import copy
import torch
import torch.nn as nn
import torch.optim as optim

from legged_gym.dwl.actor_critic_dwl import ActorCriticDWL
from legged_gym.dwl.rollout_storage_dwl import RolloutStorage

class PPODWL:
    actor_critic: ActorCriticDWL
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 teacher_path=None,
                 distill_weight=0.0,
                 distill_anneal_steps=0,
                 distill_far_distance=None,
                 distill_near_weight=0.2,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.num_short_obs = self.actor_critic.num_short_obs

        # Teacher-action distillation: a frozen copy of the parent policy
        # whose mean actions anchor the student so PPO shaping (e.g. detour
        # penalty for SPL) cannot drift the policy away from the accepted
        # model's behavior (which every unconstrained retrain did).
        self.teacher = None
        self.distill_weight = float(distill_weight)
        self.distill_anneal_steps = int(distill_anneal_steps)
        self.distill_step = 0
        self.distill_far_distance = (
            float(distill_far_distance) if distill_far_distance is not None else None
        )
        self.distill_near_weight = float(distill_near_weight)
        self.num_single_obs = self.actor_critic.num_proprio_obs
        if teacher_path:
            teacher_path = str(teacher_path).replace(
                "{LEGGED_GYM_ROOT_DIR}", "/home/jason/SphericalRobot_LeggedGym-master-new-map"
            )
            # The teacher is always the accepted DWL-CNN policy (e.g. uniform
            # 4150), regardless of the student architecture.  This keeps the
            # action anchor meaningful for cross-architecture distillation
            # (SRU student, DWL teacher): building the teacher as a copy of
            # an SRU student would anchor to random weights.
            from legged_gym.dwl.actor_critic_dwl import ActorCriticDWL

            teacher = ActorCriticDWL(
                self.actor_critic.num_short_obs,
                self.actor_critic.num_proprio_obs,
                self.actor_critic.num_critic_obs,
                self.actor_critic.num_actions,
                in_channels=getattr(self.actor_critic, "in_channels", 20),
                kernel_size=[3, 2],
                filter_size=[16, 8],
                stride_size=[1, 1],
                lh_output_dim=16,
                actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[512, 256, 128],
                activation="elu",
                init_noise_std=0.3,
                min_noise_std=0.15,
                max_noise_std=0.3,
            )
            state = torch.load(teacher_path, map_location=self.device)
            teacher_state = state["model_state_dict"]
            # Only load keys with matching shapes; report skipped tensors.
            filtered = {
                k: v
                for k, v in teacher_state.items()
                if k in teacher.state_dict()
                and teacher.state_dict()[k].shape == v.shape
            }
            skipped = len(teacher_state) - len(filtered)
            teacher.load_state_dict(filtered, strict=False)
            teacher.to(self.device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            self.teacher = teacher
            print(
                f"[distill] teacher (DWL-CNN) loaded from {teacher_path} "
                f"({skipped} tensors skipped for shape mismatch)",
                flush=True,
            )

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, None, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Teacher-action distillation (anchor against drift).
                if self.teacher is not None:
                    weight = self.distill_weight
                    if self.distill_anneal_steps > 0:
                        # Anneal the anchor from full weight down to 40% so the
                        # student can eventually refine beyond the teacher.
                        progress = min(1.0, self.distill_step / float(self.distill_anneal_steps))
                        weight = self.distill_weight * (1.0 - 0.6 * progress)
                    with torch.no_grad():
                        self.teacher.act(
                            obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
                        )
                        teacher_mu = self.teacher.action_mean
                    # State-dependent anchoring: strongly preserve teacher
                    # actions far from the target, relax near the target so
                    # the student can refine approach/braking behavior.
                    if getattr(self, "distill_far_distance", None) is not None:
                        frame = obs_batch.reshape(obs_batch.shape[0], -1, self.num_single_obs)[:, -1, :]
                        dist = (frame[:, 0:2] - frame[:, 2:4]).norm(dim=1)
                        far = (dist > self.distill_far_distance).float().unsqueeze(1)
                        per_sample = far * 1.0 + (1.0 - far) * self.distill_near_weight
                        distill_loss = torch.mean(
                            torch.square(mu_batch - teacher_mu) * per_sample
                        )
                    else:
                        distill_loss = torch.mean(torch.square(mu_batch - teacher_mu))
                    loss = loss + weight * distill_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.distill_step += 1

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss

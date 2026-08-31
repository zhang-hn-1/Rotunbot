
import time
import os
import re
from collections import deque

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist

from legged_gym.dwl.ppo_dwl import PPODWL
from legged_gym.dwl.actor_critic_dwl import ActorCriticDWL
from legged_gym.dwl.actor_critic_depth import ActorCriticDepth
from legged_gym.dwl.actor_critic_depth_local import ActorCriticDepthLocal
from legged_gym.dwl.actor_critic_direct_velocity import ActorCriticDirectVelocity
from rsl_rl.env import VecEnv


class DWLOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.action_repeat = 1
        self.primitive_gamma = None
        if bool(getattr(self.env.cfg.env, "high_level_action_timing_enabled", False)):
            from legged_gym.navigation.high_level_action_timing import derive_action_repeat

            self.action_repeat = derive_action_repeat(
                self.env.dt,
                self.env.cfg.commands.upper_level_command_frequency_hz,
            )
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCriticDWL
        actor_critic: ActorCriticDWL = actor_critic_class(
            self.env.num_short_obs, self.env.num_single_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPODWL
        self.alg: PPODWL = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.primitive_gamma = float(self.alg.gamma)
        if self.action_repeat > 1:
            self.alg.gamma = self.primitive_gamma ** self.action_repeat
            self.alg.lam = float(self.alg.lam) ** self.action_repeat
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        # Keep roughly 100 completed episodes globally in DDP diagnostics.
        self.rolling_window_per_rank = max(
            1, (100 + self.world_size - 1) // self.world_size
        )

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def _step_high_level(self, actions):
        """Hold one policy action while collecting one macro transition."""
        if self.action_repeat == 1:
            return self.env.step(actions)
        from legged_gym.navigation.high_level_action_timing import MacroStepAccumulator

        accumulator = MacroStepAccumulator(
            self.env.num_envs,
            self.action_repeat,
            self.primitive_gamma,
            device=self.device,
        )
        active = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        last_obs = last_privileged = last_infos = None
        terminal_infos = None
        with torch.inference_mode():
            for primitive_index in range(self.action_repeat):
                primitive_actions = torch.where(
                    active.unsqueeze(1), actions, torch.zeros_like(actions)
                )
                last_obs, last_privileged, rewards, dones, infos = self.env.step(
                    primitive_actions
                )
                dones = dones.flatten().bool()
                timeouts = infos.get(
                    "time_outs",
                    torch.zeros_like(dones, dtype=torch.bool),
                ).flatten().bool()
                active_before = accumulator.add(
                    rewards,
                    dones,
                    timeouts,
                    self.alg.transition.values,
                    primitive_index,
                )
                if torch.any(active_before & dones):
                    terminal_infos = infos
                active = ~accumulator.dones
                last_infos = infos
        result = accumulator.result()
        macro_infos = dict(last_infos or {})
        if terminal_infos is not None:
            macro_infos.update(terminal_infos)
        macro_infos["time_outs"] = torch.zeros_like(result.dones)
        macro_infos["timeout_bootstrap"] = result.timeout_bootstrap
        macro_infos["macro_action_repeat"] = self.action_repeat
        return (
            last_obs,
            last_privileged,
            result.rewards.squeeze(1),
            result.dones.squeeze(1),
            macro_infos,
        )
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        episode_metric_buffers = {
            "reward": deque(maxlen=self.rolling_window_per_rank),
            "length": deque(maxlen=self.rolling_window_per_rank),
        }
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            diagnostic_sums = {
                "path_distance": torch.zeros((), device=self.device),
                "base_speed": torch.zeros((), device=self.device),
                "collision_rate": torch.zeros((), device=self.device),
            }
            goal_sampling_counts = {
                "near": torch.zeros((), device=self.device),
                "mid": torch.zeros((), device=self.device),
                "far": torch.zeros((), device=self.device),
            }
            # Count the initial goal assigned to every environment once at the
            # beginning of a run (or after resuming from a checkpoint).
            if (
                it == self.current_learning_iteration
                and hasattr(self.env, "maze_goal_sampling_bin")
            ):
                initial_bins = self.env.maze_goal_sampling_bin
                for bin_id, bin_name in enumerate(("near", "mid", "far")):
                    goal_sampling_counts[bin_name] += (
                        initial_bins == bin_id
                    ).float().sum()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # env.step() returns reset observations for environments
                    # that terminate on this transition. Snapshot state
                    # diagnostics first so the next episode's initial
                    # state is not mixed into this rollout's distance and
                    # speed averages. Collision is measured after step().
                    path_dist_before = getattr(
                        self.env,
                        "maze_goal_distance",
                        getattr(self.env, "goal_dist", None),
                    )
                    base_speed_before = getattr(self.env, "base_lin_vel", None)
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self._step_high_level(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    collision_buf = getattr(
                        self.env,
                        "step_collision_buf",
                        getattr(self.env, "obstacle_collision_buf", None),
                    )
                    if path_dist_before is not None:
                        diagnostic_sums["path_distance"] += (
                            path_dist_before.detach().mean()
                        )
                    if base_speed_before is not None:
                        diagnostic_sums["base_speed"] += torch.linalg.vector_norm(
                            base_speed_before.detach(), dim=1
                        ).mean()
                    if collision_buf is not None:
                        diagnostic_sums["collision_rate"] += (
                            collision_buf.detach().float().mean()
                        )

                    done_mask = dones.flatten() > 0
                    if done_mask.any() and hasattr(
                        self.env, "maze_goal_sampling_bin"
                    ):
                        # reset_idx has assigned the next episode's goal by
                        # the time env.step returns.
                        reset_bins = self.env.maze_goal_sampling_bin[done_mask]
                        for bin_id, bin_name in enumerate(("near", "mid", "far")):
                            goal_sampling_counts[bin_name] += (
                                reset_bins == bin_id
                            ).float().sum()

                    # Book keeping is performed on every DDP rank so episode
                    # metrics can be reduced globally before rank 0 logs them.
                    # Episode info can persist in extras after a reset, so
                    # record it only when this step really has done envs.
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    if new_ids.numel() > 0 and "episode" in infos:
                        ep_infos.append(infos["episode"])
                        episode_count = infos["episode"].get(
                            "_episode_count", 1.0
                        )
                        if isinstance(episode_count, torch.Tensor):
                            episode_count = int(round(float(episode_count.item())))
                        episode_count = max(int(episode_count), 1)
                        for metric_name in episode_metric_buffers:
                            if metric_name not in infos["episode"]:
                                continue
                            metric_value = infos["episode"][metric_name]
                            if isinstance(metric_value, torch.Tensor):
                                metric_value = float(metric_value.mean().item())
                            for _ in range(episode_count):
                                episode_metric_buffers[metric_name].append(
                                    float(metric_value)
                                )
                    if new_ids.numel() > 0:
                        completed_rewards = (
                            cur_reward_sum[new_ids][:, 0].detach().cpu().tolist()
                        )
                        completed_lengths = (
                            cur_episode_length[new_ids][:, 0].detach().cpu().tolist()
                        )
                        episode_metric_buffers["reward"].extend(completed_rewards)
                        episode_metric_buffers["length"].extend(completed_lengths)
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop
            diagnostic_means = {
                key: value / float(self.num_steps_per_env)
                for key, value in diagnostic_sums.items()
            }
            if dist.is_available() and dist.is_initialized():
                diagnostic_tensor = torch.stack(
                    [diagnostic_means[key] for key in diagnostic_means]
                )
                dist.all_reduce(diagnostic_tensor, op=dist.ReduceOp.SUM)
                diagnostic_tensor.div_(dist.get_world_size())
                diagnostic_means = {
                    key: diagnostic_tensor[index]
                    for index, key in enumerate(diagnostic_means)
                }
                sampling_tensor = torch.stack(
                    [
                        goal_sampling_counts[key]
                        for key in ("near", "mid", "far")
                    ]
                )
                dist.all_reduce(sampling_tensor, op=dist.ReduceOp.SUM)
                goal_sampling_counts = {
                    key: sampling_tensor[index]
                    for index, key in enumerate(("near", "mid", "far"))
                }
            aggregated_episode_metrics = self._aggregate_episode_metrics(ep_infos)
            rolling_episode_metrics = self._aggregate_rolling_metrics(
                episode_metric_buffers
            )
            self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            sequence_metadata = getattr(self.alg, "sequence_metadata", None)
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
                # Make the event file observable even when a run is stopped
                # or resumed from a checkpoint shortly after this iteration.
                self.writer.flush()
            if self.log_dir is not None and it % self.save_interval == 0:
                self.save(
                    os.path.join(self.log_dir, 'model_{}.pt'.format(it)),
                    iteration=it + 1,
                )
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            self.writer.flush()

    def _episode_metric_names(self):
        names = {
            "success",
            "collision",
            "timeout",
            "unstable",
            "out_of_bounds",
            "terminal_goal_distance",
            "terminal_speed",
        }
        return sorted(names)

    def _aggregate_episode_metrics(self, ep_infos):
        """Reduce newly completed episode metrics across all DDP ranks."""
        aggregated = {}
        for key in self._episode_metric_names():
            local_sum = torch.zeros((), device=self.device)
            local_count = torch.zeros((), device=self.device)
            for ep_info in ep_infos:
                if key not in ep_info:
                    continue
                if isinstance(ep_info[key], torch.Tensor):
                    metric_value = ep_info[key].to(self.device).float().mean()
                else:
                    metric_value = torch.tensor(
                        float(ep_info[key]), device=self.device
                    )
                episode_count = ep_info.get("_episode_count", 1.0)
                if isinstance(episode_count, torch.Tensor):
                    episode_count = episode_count.to(self.device).float().mean()
                else:
                    episode_count = torch.tensor(
                        float(episode_count), device=self.device
                    )
                local_sum += metric_value * episode_count
                local_count += episode_count
            if dist.is_available() and dist.is_initialized():
                aggregate = torch.stack((local_sum, local_count))
                dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
                local_sum, local_count = aggregate.unbind()
            if float(local_count.item()) > 0.0:
                aggregated[key] = local_sum / local_count

        # Keep the denominator visible.  A per-iteration success rate based on
        # one or two completed episodes is numerically valid but statistically
        # very noisy, so the log must expose how many episodes produced it.
        local_episode_count = torch.zeros((), device=self.device)
        local_success_count = torch.zeros((), device=self.device)
        for ep_info in ep_infos:
            episode_count = ep_info.get("_episode_count", 1.0)
            if isinstance(episode_count, torch.Tensor):
                episode_count = episode_count.to(self.device).float().mean()
            else:
                episode_count = torch.tensor(
                    float(episode_count), device=self.device
                )
            local_episode_count += episode_count
            if "success" in ep_info:
                success_value = ep_info["success"]
                if isinstance(success_value, torch.Tensor):
                    success_value = success_value.to(self.device).float().mean()
                else:
                    success_value = torch.tensor(
                        float(success_value), device=self.device
                    )
                local_success_count += success_value * episode_count
        if dist.is_available() and dist.is_initialized():
            count_tensor = torch.stack((local_episode_count, local_success_count))
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            local_episode_count, local_success_count = count_tensor.unbind()
        if float(local_episode_count.item()) > 0.0:
            aggregated["completed_count"] = local_episode_count
            aggregated["success_count"] = local_success_count
        return aggregated

    def _aggregate_rolling_metrics(self, metric_buffers):
        """Reduce the last-100-episode metrics across all DDP ranks."""
        aggregated = {}
        for key in sorted(metric_buffers):
            values = metric_buffers[key]
            stats = torch.tensor(
                [sum(values), float(len(values))],
                dtype=torch.float32,
                device=self.device,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            if float(stats[1].item()) > 0.0:
                aggregated[key] = stats[0] / stats[1]
        return aggregated

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs * self.world_size
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs.get('aggregated_episode_metrics'):
            for key, value in sorted(locs['aggregated_episode_metrics'].items()):
                if key in ("completed_count", "success_count"):
                    continue
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
            completed_count = locs['aggregated_episode_metrics'].get(
                "completed_count"
            )
            success_count = locs['aggregated_episode_metrics'].get(
                "success_count"
            )
            if completed_count is not None and success_count is not None:
                completed_count = float(
                    completed_count.item()
                    if isinstance(completed_count, torch.Tensor)
                    else completed_count
                )
                success_count = float(
                    success_count.item()
                    if isinstance(success_count, torch.Tensor)
                    else success_count
                )
                self.writer.add_scalar(
                    'Episode/completed_count', completed_count, locs['it']
                )
                self.writer.add_scalar(
                    'Episode/success_count', success_count, locs['it']
                )
                ep_string += (
                    f"{f'Completed episodes:':>{pad}} "
                    f"{int(completed_count)}  "
                    f"{f'Success episodes:':>{pad}} "
                    f"{int(success_count)}\n"
                )
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            * self.world_size
            / (locs['collection_time'] + locs['learn_time'])
        )

        curriculum_string = ''
        get_curriculum_status = getattr(
            self.env, 'get_tracking_curriculum_status', None
        )
        curriculum_status = (
            get_curriculum_status() if callable(get_curriculum_status) else None
        )
        if curriculum_status:
            stage = int(curriculum_status['stage'])
            max_stage = int(curriculum_status['max_stage'])
            stage_name = curriculum_status['stage_name']
            batch_pass_rate = float(curriculum_status['batch_pass_rate'])
            ema_pass_rate = float(curriculum_status['ema_pass_rate'])
            updates = int(curriculum_status['updates'])
            min_updates = int(curriculum_status['min_updates'])
            pass_threshold = float(curriculum_status['pass_threshold'])
            detail_ema = curriculum_status.get('detail_ema', {})
            self.writer.add_scalar('Curriculum/stage', stage, locs['it'])
            self.writer.add_scalar(
                'Curriculum/batch_pass_rate', batch_pass_rate, locs['it']
            )
            self.writer.add_scalar(
                'Curriculum/ema_pass_rate', ema_pass_rate, locs['it']
            )
            self.writer.add_scalar('Curriculum/updates', updates, locs['it'])
            for detail_name, detail_value in detail_ema.items():
                self.writer.add_scalar(
                    'Curriculum/detail_' + detail_name,
                    float(detail_value),
                    locs['it'],
                )
            curriculum_string += (
                f"{f'Tracking curriculum stage:':>{pad}} "
                f"{stage}/{max_stage} ({stage_name})\n"
                f"{f'Tracking pass rate (batch/EMA):':>{pad}} "
                f"{batch_pass_rate:.1%} / {ema_pass_rate:.1%}\n"
            )
            if detail_ema:
                curriculum_string += (
                    f"{f'Tracking channel EMA (v/w/vy):':>{pad}} "
                    f"{detail_ema['speed']:.1%} / "
                    f"{detail_ema['turn']:.1%} / "
                    f"{detail_ema['lateral']:.1%}\n"
                    f"{f'Tracking mode EMA (stop/straight):':>{pad}} "
                    f"{detail_ema['stop']:.1%} / "
                    f"{detail_ema['straight']:.1%}\n"
                    f"{f'Tracking curve EMA (fwd/rev/L/R):':>{pad}} "
                    f"{detail_ema['forward_curve']:.1%} / "
                    f"{detail_ema['reverse_curve']:.1%} / "
                    f"{detail_ema['left']:.1%} / "
                    f"{detail_ema['right']:.1%}\n"
                )
            if stage < max_stage:
                curriculum_string += (
                    f"{f'Tracking stage progress:':>{pad}} "
                    f"{updates}/{min_updates} updates, "
                    f"required EMA {pass_threshold:.1%}\n"
                )
            else:
                curriculum_string += (
                    f"{f'Tracking stage progress:':>{pad}} "
                    f"final stage, {updates} measured updates\n"
                )

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        sequence_metadata = locs.get("sequence_metadata") or {}
        for key in (
            "sequence_length",
            "sequence_batch_size",
            "number_of_sequences",
        ):
            if key in sequence_metadata:
                self.writer.add_scalar(
                    "Recurrent/" + key,
                    sequence_metadata[key],
                    locs['it'],
                )
        diagnostic_labels = {
            "path_distance": "Diagnostics/rollout_mean_path_distance",
            "base_speed": "Diagnostics/mean_base_speed",
            "collision_rate": "Diagnostics/mean_collision_rate",
        }
        diagnostic_values = locs.get("diagnostic_means", {})
        for key, label in diagnostic_labels.items():
            if key in diagnostic_values:
                value = diagnostic_values[key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.writer.add_scalar(label, value, locs['it'])
        sampling_counts = locs.get("goal_sampling_counts", {})
        sampling_string = ""
        if sampling_counts:
            count_values = {
                key: float(value.item() if isinstance(value, torch.Tensor) else value)
                for key, value in sampling_counts.items()
            }
            total_goal_samples = sum(count_values.values())
            if total_goal_samples > 0.0:
                fractions = {
                    key: count_values[key] / total_goal_samples
                    for key in ("near", "mid", "far")
                }
                for key in ("near", "mid", "far"):
                    self.writer.add_scalar(
                        "Diagnostics/goal_samples_" + key,
                        count_values[key],
                        locs['it'],
                    )
                    self.writer.add_scalar(
                        "Diagnostics/goal_sample_fraction_" + key,
                        fractions[key],
                        locs['it'],
                    )
                sampling_string = (
                    f"{'Goal samples near/mid/far:':>{pad}} "
                    f"{int(count_values['near'])}/"
                    f"{int(count_values['mid'])}/"
                    f"{int(count_values['far'])}  "
                    f"({fractions['near']:.1%}/"
                    f"{fractions['mid']:.1%}/"
                    f"{fractions['far']:.1%})\n"
                )
        rolling_metrics = locs.get('rolling_episode_metrics', {})
        has_episode_stats = 'reward' in rolling_metrics and 'length' in rolling_metrics
        if has_episode_stats:
            mean_reward = float(rolling_metrics['reward'])
            mean_episode_length = float(rolling_metrics['length'])
            self.writer.add_scalar(
                'Train/recent_mean_reward_100', mean_reward, locs['it']
            )
            self.writer.add_scalar(
                'Train/recent_mean_episode_length_100',
                mean_episode_length,
                locs['it'],
            )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if has_episode_stats:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                            f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                            f"""{'Recent mean reward (<=100 ep):':>{pad}} {mean_reward:.2f}\n"""
                            f"""{'Recent episode length (<=100 ep):':>{pad}} {mean_episode_length:.2f}\n""")
            if diagnostic_values:
                log_string += (
                    f"""{'Mean rollout path distance:':>{pad}} """
                    f"""{float(diagnostic_values['path_distance']):.3f}\n"""
                    f"""{'Mean base speed:':>{pad}} """
                    f"""{float(diagnostic_values['base_speed']):.3f}\n"""
                    f"""{'Mean collision rate:':>{pad}} """
                    f"""{float(diagnostic_values['collision_rate']):.3f}\n"""
                )
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            if diagnostic_values:
                log_string += (
                    f"""{'Mean rollout path distance:':>{pad}} """
                    f"""{float(diagnostic_values['path_distance']):.3f}\n"""
                    f"""{'Mean base speed:':>{pad}} """
                    f"""{float(diagnostic_values['base_speed']):.3f}\n"""
                    f"""{'Mean collision rate:':>{pad}} """
                    f"""{float(diagnostic_values['collision_rate']):.3f}\n"""
                )

        log_string += ep_string
        log_string += sampling_string
        log_string += curriculum_string
        remaining_iterations = max(
            0,
            self.current_learning_iteration
            + locs['num_learning_iterations']
            - locs['it']
            - 1,
        )
        completed_iterations = max(
            1, locs['it'] - self.current_learning_iteration + 1
        )
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / completed_iterations * remaining_iterations:.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None, iteration=None):
        env_state = None
        get_env_state = getattr(self.env, "get_checkpoint_state", None)
        if get_env_state is not None:
            env_state = get_env_state()
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration if iteration is None else iteration,
            'infos': infos,
            'env_state': env_state,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        env_state = loaded_dict.get('env_state')
        required_env_state_version = self.cfg.get(
            'load_optimizer_env_state_version', None
        )
        compatible_env_state = isinstance(env_state, dict)
        if compatible_env_state and required_env_state_version is not None:
            compatible_env_state = (
                env_state.get('checkpoint_state_version')
                == required_env_state_version
            )
        task_continuation = (
            self.cfg.get('load_optimizer_when_env_state', False)
            and compatible_env_state
        )
        load_optimizer = load_optimizer and (
            self.cfg.get('load_optimizer', True) or task_continuation
        )
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        elif 'optimizer_state_dict' in loaded_dict:
            print(
                "Optimizer state not loaded; using configured learning rate "
                f"{self.alg.learning_rate:.2e}"
            )

        # Older intermediate checkpoints were saved before the runner's
        # current iteration was updated.  Recover the intended iteration from
        # their filename when the stored value is stale.
        loaded_iteration = int(loaded_dict.get('iter', 0))
        match = re.search(r"model_(\d+)\.pt$", str(path))
        if match:
            loaded_iteration = max(loaded_iteration, int(match.group(1)))
        self.current_learning_iteration = loaded_iteration

        set_env_state = getattr(self.env, "set_checkpoint_state", None)
        if set_env_state is not None:
            set_env_state(loaded_dict.get('env_state'))
        return loaded_dict.get('infos')

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_inference_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate

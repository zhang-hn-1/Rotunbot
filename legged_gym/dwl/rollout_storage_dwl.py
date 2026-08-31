# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

import torch

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.next_proprio_obs = None
        
        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, num_single_obs=None, device='cpu', recurrent=False):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        if num_single_obs is not None:
            self.next_proprio_obs = torch.zeros(num_transitions_per_env, num_envs, num_single_obs, device=self.device)
            
        self.num_single_obs = num_single_obs
        self.recurrent = bool(recurrent)
        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None
        self.last_sequence_metadata = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        if self.num_single_obs is not None:
            self.next_proprio_obs[self.step].copy_(transition.next_proprio_obs)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None:
            return
        if isinstance(hidden_states, tuple):
            if not hidden_states or hidden_states[0] is None:
                return
            actor_hidden = hidden_states[0]
        else:
            actor_hidden = hidden_states
        hid_a = actor_hidden if isinstance(actor_hidden, tuple) else (actor_hidden,)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])


    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.recurrent:
            yield from self._recurrent_mini_batch_generator(num_mini_batches, num_epochs)
            return
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        if self.num_single_obs is not None:
            next_proprio_obs = self.next_proprio_obs.flatten(0, 1)
            rewards = self.rewards.flatten(0, 1)
        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                if self.num_single_obs is not None:
                    next_proprio_obs_batch = next_proprio_obs[batch_idx]
                    rewards_batch = rewards[batch_idx]
                    yield next_proprio_obs_batch, rewards_batch, obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                        old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None
                else:
                    yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                        old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None

    def _recurrent_mini_batch_generator(self, num_mini_batches, num_epochs):
        """Yield full chronological sequences for recurrent PPO updates.

        Environments are the independent sequence units.  A mini-batch may
        contain several environments, but it never shuffles time within one
        environment.  The mask at time ``t`` resets the carried state when
        the preceding transition ended an episode.
        """
        if self.num_envs < num_mini_batches:
            raise ValueError("recurrent PPO requires num_envs >= num_mini_batches")
        mini_batch_size = self.num_envs // num_mini_batches
        usable_envs = num_mini_batches * mini_batch_size
        indices = torch.randperm(usable_envs, requires_grad=False, device=self.device)
        observations = self.observations
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations
        else:
            critic_observations = observations
        if self.saved_hidden_states_a is None:
            raise RuntimeError("recurrent rollout is missing initial hidden states")
        done = self.dones.squeeze(-1).float()
        self.last_sequence_metadata = {
            "sequence_length": int(self.num_transitions_per_env),
            "sequence_batch_size": int(mini_batch_size),
            "number_of_sequences": int(usable_envs),
            "hidden_shape": list(self.saved_hidden_states_a[0].shape[1:]),
        }
        for _ in range(num_epochs):
            for batch_index in range(num_mini_batches):
                start = batch_index * mini_batch_size
                end = (batch_index + 1) * mini_batch_size
                env_indices = indices[start:end]
                masks = torch.ones(
                    self.num_transitions_per_env,
                    mini_batch_size,
                    device=self.device,
                )
                if self.num_transitions_per_env > 1:
                    masks[1:] = 1.0 - done[:-1, env_indices]
                hidden = self.saved_hidden_states_a[0][0, env_indices]
                yield (
                    observations[:, env_indices],
                    critic_observations[:, env_indices],
                    self.actions[:, env_indices],
                    self.values[:, env_indices],
                    self.advantages[:, env_indices],
                    self.returns[:, env_indices],
                    self.actions_log_prob[:, env_indices],
                    self.mu[:, env_indices],
                    self.sigma[:, env_indices],
                    (hidden, None),
                    masks,
                )

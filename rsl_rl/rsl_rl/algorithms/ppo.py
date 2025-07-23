#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

import copy
import random

class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
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
        device="cpu",
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.AdamW(self.actor_critic.parameters(), lr=learning_rate)
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
        self.ci_max = None

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, reward_components):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, reward_components, self.device
        )

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
        #self.transition.constraints = constraints.detach()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        reward_components = self.transition.rewards.shape[1]
        # Bootstrapping on time outs

        if "time_outs" in infos:
            self.transition.rewards += self.gamma * self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device)
            
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        #self.ci_max = self.storage.adjust_rewards(self.ci_max, pi_curr)
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def apply_pcgrad(self, objective_grads):
        num_objectives = len(objective_grads)

        # Flatten gradients
        flat_grads = torch.stack([
            torch.cat([g.flatten() for g in grad_list])
            for grad_list in objective_grads
        ])  # [num_objectives, param_dim]

        # Dot product matrix
        dot_prods = flat_grads @ flat_grads.T
        norms_sq = torch.diag(dot_prods).unsqueeze(0) + 1e-10  # [1, num_objectives]

        # Create masks for conflicts (dot < 0)
        conflict_mask = dot_prods < 0

        # Randomize order of projections
        rand_perm = [torch.randperm(num_objectives) for _ in range(num_objectives)]
        projected_grads = flat_grads.clone()

        for i in range(num_objectives):
            j_indices = rand_perm[i][rand_perm[i] != i]
            g_i = projected_grads[i]

            for j in j_indices:
                if conflict_mask[i, j]:
                    proj_coeff = dot_prods[i, j] / norms_sq[0, j]
                    g_i = g_i - proj_coeff * flat_grads[j]
            projected_grads[i] = g_i

        # Average and reshape
        combined_flat = projected_grads.mean(dim=0)

        combined_grads = []
        start = 0
        for g in objective_grads[0]:
            numel = g.numel()
            combined_grads.append(combined_flat[start:start+numel].view_as(g))
            start += numel

        return combined_grads


    def update(self):
        mean_value_loss = 0
        mean_component_value_loss = torch.zeros((self.actor_critic.num_reward_components), device=self.device)
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            component_advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
                #critic_obs_batch, masks=None, hidden_states=None
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            ratio = ratio.unsqueeze(1)  # shape: [B, 1]
            component_advantages_batch = component_advantages_batch  # shape: [B, C]

            surrogate_per_component = -component_advantages_batch * ratio  # shape: [B, C]
            surrogate_per_component_clipped = -component_advantages_batch * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )

            component_surrogate_losses = torch.max(
                surrogate_per_component, surrogate_per_component_clipped
            )  # [B, C]

            mean_component_surrogate_loss = component_surrogate_losses.mean(dim=0)  # [C]
            surrogate_loss = mean_component_surrogate_loss.sum()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                dim_use = 2 if len(value_losses.shape) > 2 else 1
                value_loss = torch.max(value_losses.sum(dim=dim_use), value_losses_clipped.sum(dim=dim_use)).mean()
            else:
                assert False # not implemented
                value_loss = (returns_batch - value_batch).pow(2).sum(dim=2).mean()
                mean_component_value_loss += (returns_batch - value_batch).pow(2).sum(dim=2).mean(dim=0).mean(dim=0)

            # More efficient per-component gradient computation
            self.optimizer.zero_grad()
            
            # Compute gradients for all components in one backward pass
            # Create a tensor of ones for each component to compute gradients
            num_components = mean_component_surrogate_loss.size(0)
            component_grads = []
            
            for i in range(num_components):
                # Use autograd.grad for more efficient gradient computation
                grads = torch.autograd.grad(
                    outputs=mean_component_surrogate_loss[i],
                    inputs=self.actor_critic.parameters(),
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )
                # Convert None gradients to zero tensors
                grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, self.actor_critic.parameters())]
                component_grads.append(grads)

            # Apply PCGrad
            projected_grads = self.apply_pcgrad(component_grads)

            # Assign PCGrad-updated gradients to model
            for param, g in zip(self.actor_critic.parameters(), projected_grads):
                if param.requires_grad:
                    param.grad = g

            # Value loss + entropy (backward separately and add)
            value_loss.backward(retain_graph=True)
            if self.entropy_coef > 0:
                (-self.entropy_coef * entropy_batch.mean()).backward()

            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()


            '''loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            '''
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_component_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_component_value_loss

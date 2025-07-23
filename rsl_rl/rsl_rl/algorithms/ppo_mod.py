#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic, ActorCriticModulator
from rsl_rl.storage import RolloutStorageMod
import time
import math

class PPOMod:
    actor_critic: ActorCriticModulator

    def __init__(
        self,
        actor_critic,
        modulation_size,modulation_params_perc,
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
        self.transition = RolloutStorageMod.Transition()

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
        self.base_ac = None
        self.ci_max = None
        self.modulation_size =modulation_size
        self.modulation_params_perc = modulation_params_perc

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, reward_components, context_shape):
        self.storage = RolloutStorageMod(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, reward_components, context_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, base_ac, weight_to_add, bias_to_add, max_torques):
        if self.base_ac == None:
            self.base_ac = base_ac
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.forward_modulated(base_ac, obs, weight_to_add["actor"], bias_to_add["actor"]).detach()
        self.transition.context = max_torques
        #self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate_modulated(base_ac, critic_obs, weight_to_add["critic"], bias_to_add["critic"]).detach()
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

    def compute_returns(self, last_critic_obs, base_ac, weight_to_add, bias_to_add):
        last_values = self.actor_critic.evaluate_modulated(base_ac, last_critic_obs, weight_to_add["critic"], bias_to_add["critic"]).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_modulation(self, context, residual_net):
        """
        Compute modulation tensors (weight deltas and biases) for a given residual network.
        This function assumes that the network is composed solely of nn.Linear layers.
        
        Args:
            context (Tensor): The input tensor (e.g. context_batch).
            residual_net (nn.Module): The residual network module.

        Returns:
            Tuple[List[Tensor], List[Tensor]]: The computed weight deltas and biases.
        """

        weight_deltas = []
        biases = []
        res_input_actor = context.clone()#.reshape(-1,1)

        for idx, res_layer_actor in enumerate(residual_net):
            if idx == 0 or not "Linear" in str(type(res_layer_actor)): # either activation or the linear layer
                res_input_actor = res_layer_actor(res_input_actor)
                continue

            delta = res_input_actor.unsqueeze(1) * res_layer_actor.weight.unsqueeze(0)
            if len(weight_deltas) < 3:
                delta = delta[:,:math.ceil(delta.shape[1]/self.modulation_size),:math.ceil(delta.shape[2]/self.modulation_size)]
            else:
                delta = delta[:,:,:math.ceil(delta.shape[2]/self.modulation_size)]

            mod_param_limit = int(delta.shape[-1] * self.modulation_params_perc)
            delta = delta[:, :, :mod_param_limit]

            weight_deltas.append(delta)
            if len(biases) < 3:
                bias = res_layer_actor.bias[:math.ceil(res_layer_actor.bias.shape[0]/self.modulation_size)]
            else:
                bias = res_layer_actor.bias 
            mod_bias_limit = int(bias.shape[0] * self.modulation_params_perc)
            bias = bias[:mod_bias_limit]
            biases.append(bias)
            res_input_actor = res_layer_actor(res_input_actor)

        return weight_deltas, biases

    def update(self, new_actor_critic, it):
        residual_actor = new_actor_critic.actor
        residual_critic = new_actor_critic.critic
                
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
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            context_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            actor_weight_deltas, actor_biases = self.compute_modulation(context_batch, residual_actor)
            #if it < 50:
            #    actor_weight_deltas = [x*0 for x in actor_weight_deltas]
            #    actor_biases = [x*0 for x in actor_biases]
            self.actor_critic.forward_modulated(self.base_ac, obs_batch, actor_weight_deltas, actor_biases)
            del actor_weight_deltas, actor_biases
            critic_weight_deltas, critic_biases = self.compute_modulation(context_batch, residual_critic)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate_modulated(
                self.base_ac, critic_obs_batch,critic_weight_deltas, critic_biases)
            del critic_weight_deltas, critic_biases
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            #actions_log_prob_batch.requires_grad = True
            #old_actions_log_prob_batch.requires_grad = True
            #self.actor_critic.forward_modulated(self.base_ac, obs_batch, actor_weight_deltas, actor_biases)
            #del actor_weight_deltas, actor_biases
            #critic_weight_deltas, critic_biases = self.compute_modulation(context_batch, residual_critic)
            #actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            #value_batch = self.actor_critic.evaluate_modulated(
            #    self.base_ac, critic_obs_batch,critic_weight_deltas, critic_biases
            #)
            #del critic_weight_deltas, critic_biases
            #mu_batch = self.actor_critic.action_mean
            #sigma_batch = self.actor_critic.action_std
            #entropy_batch = self.actor_critic.entropy

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
                    #print(f"KL: {kl_mean}")
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            #print("Advantages:", torch.mean(advantages_batch), torch.std(advantages_batch))
            #print("ratio:", torch.mean(ratio))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

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

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()

            '''total_norm = 0.0
            for name, param in self.actor_critic.actor.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.detach().data.norm(2)  # L2 norm
                    total_norm += param_norm.item() ** 2
                    print(f"{name}: {param_norm.item():.6f}")
            total_norm = total_norm ** 0.5
            print(f"Total gradient norm: {total_norm:.4f}")
            for name, param in self.actor_critic.actor.named_parameters():
                if not param.requires_grad:
                    print(f"{name} is frozen!")'''



            #for param in self.base_ac.parameters():
            #    pass
                #if param.grad is not None:
                #    assert False
            
            '''for name,param in self.actor_critic.named_parameters():
                if param.grad is not None:
                    print(f"{name}: {param.grad.mean().item()}, std: {param.grad.std().item()}")
                else:
                    print(f"{name} has no gradient!")'''
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            '''actor_lr_scale = min(1.0, max(0.0, (it - 10) / 40))  # ramps from 0 at it=10 to 1 at it=50
            for param in self.actor_critic.actor.parameters():
                if param.grad is not None:
                    param.grad.mul_(actor_lr_scale)'''
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_component_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_component_value_loss

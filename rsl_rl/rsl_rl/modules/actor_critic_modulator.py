#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F

    
class ActorCriticModulator(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_reward_components,
        context_dim=1,
        modulation_size=1.0, 
        modulation_params_perc=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0, # can be a trainable tensor
        
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.num_reward_components = num_reward_components

        activation = get_activation(activation)

        mlp_input_dim_a = int(num_actor_obs*modulation_size)
        mlp_input_dim_c = int(num_critic_obs*modulation_size)

        # Policy
        actor_hidden_dims = [int(256*modulation_size), int(128*modulation_size),int(64*modulation_size)]
        critic_hidden_dims = [int(256*modulation_size), int(128*modulation_size),int(64*modulation_size)]
        actor_layers = []
         
        actor_layers.append(nn.Linear(context_dim,mlp_input_dim_a))
        actor_layers.append(activation)
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)

        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(context_dim,mlp_input_dim_c))
        critic_layers.append(activation)

        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], num_reward_components))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        # Action noised
        self.std = nn.Parameter(torch.tensor(init_noise_std), requires_grad=True)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        self.actor.apply(self.init_weights_near_zero)
        self.critic.apply(self.init_weights_near_zero)

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    def init_weights_near_zero(self,m):
        """Initialize weights near 0 with small noise and biases to 0."""
        if isinstance(m, nn.Linear):
            # Initialize weights close to 1 with small random noise
            with torch.no_grad():
                m.weight.copy_(torch.zeros_like(m.weight) + 0.02 * torch.randn_like(m.weight))
            # Set bias to zero
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def modulated_linear(self, x, base_layer, delta_weight, delta_bias):
        """
        Modulates only a portion of the base_layer using delta_weight and delta_bias.
        Assumes delta_weight.shape = (batch_size, mod_out_features, mod_in_features)
        and delta_bias.shape = (batch_size, mod_out_features)
        """
        batch_size = x.size(0)
        #full_out_features = base_layer.weight.size(0)
        #full_in_features = base_layer.weight.size(1)

        mod_out_features = delta_weight.size(1)
        mod_in_features = delta_weight.size(2)

        # Create base_weight and base_bias tensors (broadcasted across batch)
        base_weight = base_layer.weight.unsqueeze(0).expand(batch_size, -1, -1).clone()
        base_bias = base_layer.bias.unsqueeze(0).expand(batch_size, -1).clone()

        # Apply modulation only to first part of the weights and biases
        base_weight[:, :mod_out_features, :mod_in_features] += delta_weight
        base_bias[:, :delta_bias.shape[0]] += delta_bias

        # x: (batch_size, in_features), reshape to (batch_size, 1, in_features)
        x_unsqueezed = x.unsqueeze(1)
        effective_weight_t = base_weight.transpose(1, 2)  # (batch_size, in_features, out_features)

        out = torch.bmm(x_unsqueezed, effective_weight_t).squeeze(1) + base_bias
        out[torch.isnan(out)] = 0
        return out

    def forward_modulated(self, actor_critic, x, weight_deltas, bias_deltas):
        # Assume weight_deltas and bias_deltas are lists corresponding to each linear layer in actor.
        layer_idx = 0
        actor = actor_critic.actor
        actor_layers = [layer for layer in actor.children()]
        x_copy = x.clone()
        for layer in actor_layers:
            if isinstance(layer, nn.Linear) or layer.original_name == "Linear":
                # Use modulated linear computation for this layer.
                x_copy = self.modulated_linear(x_copy, layer, weight_deltas[layer_idx], bias_deltas[layer_idx])
                layer_idx += 1
            else:
                # For activation functions or other layers, simply pass through.
                x_copy = layer(x_copy)

        x_copy[torch.isnan(x_copy)] = 0

        self.distribution = Normal(x_copy, self.std)
        
        #print("weight delta mean: ", torch.mean(weight_deltas[0]), torch.mean(weight_deltas[1]), torch.mean(weight_deltas[2]))
        return self.distribution.rsample()

    def evaluate_modulated(self, actor_critic, x, weight_deltas, bias_deltas):
        layer_idx = 0
        actor, critic = actor_critic.actor, actor_critic.critic
        critic_layers = [layer for layer in critic.children()]
        x_copy = x.clone()
        for layer in critic_layers:
            if isinstance(layer, nn.Linear) or layer.original_name == "Linear":
                # Use modulated linear computation for this layer.
                x_copy = self.modulated_linear(x_copy, layer, weight_deltas[layer_idx], bias_deltas[layer_idx])
                layer_idx += 1
            else:
                # For activation functions or other layers, simply pass through.
                x_copy = layer(x_copy)
        x_copy[torch.isnan(x_copy)] = 0
        return x_copy

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        mean[torch.isnan(mean)] = 0
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, base_ac, context, observations):
        # figure out weight and bias deltas
        weight_deltas = []
        bias_deltas = []
        res_input_orig = context.clone()#.reshape(-1,1)

        residual_layers_actor = [layer for layer in self.actor.children() if isinstance(layer, nn.Linear)]
        res_input_actor = residual_layers_actor[0](res_input_orig)

        for res_layer_actor in residual_layers_actor[1:]:
            delta_actor = res_input_actor.unsqueeze(1) * res_layer_actor.weight.unsqueeze(0)
            weight_deltas.append(delta_actor)
            bias_deltas.append(res_layer_actor.bias)
            res_input_actor = res_layer_actor(res_input_actor)
        
        layer_idx = 0
        actor = base_ac.actor
        actor_layers = [layer for layer in actor.children()]
        x_copy = observations.clone()
        for layer in actor_layers:
            if isinstance(layer, nn.Linear) or layer.original_name == "Linear":
                # Use modulated linear computation for this layer.
                x_copy = self.modulated_linear(x_copy, layer, weight_deltas[layer_idx], bias_deltas[layer_idx])
                layer_idx += 1
            else:
                # For activation functions or other layers, simply pass through.
                x_copy = layer(x_copy)
        return x_copy
    
    '''def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def evaluate(self, critic_observations, **kwargs):
        values = self.critic(critic_observations)
        return values'''


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

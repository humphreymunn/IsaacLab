#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from copy import deepcopy
import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import torch.nn as nn
import math
import rsl_rl
from rsl_rl.algorithms import PPO, PPOMod
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticModulator, EmpiricalNormalization
from rsl_rl.utils import store_code_state


class ModulationRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu", multihead=False, base_path=None,modulation_size=1.0,modulation_params_perc=1.0):
        
        self.device = device
        self.env = env
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]

        self.base_ac = torch.jit.load(base_path)
        for param in self.base_ac.parameters():
            param.requires_grad = False
        self.base_actor, self.base_critic = self.base_ac.actor.to(self.device), self.base_ac.critic.to(self.device)

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs

        self.reward_components = 1 if not multihead else env.env.reward_components
        self.policy_cfg = train_cfg["policy"]
        self.context_dim = self.env.env.context_size
        self.new_actor_critic: ActorCriticModulator = ActorCriticModulator(
            num_obs, num_critic_obs, self.env.num_actions, self.reward_components, self.context_dim, modulation_size, modulation_params_perc, **self.policy_cfg
        ).to(self.device)

        self.modulation_size = modulation_size
        self.modulation_params_perc = modulation_params_perc

        self.alg_cfg["class_name"] = "PPOMod"
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: PPOMod = alg_class(self.new_actor_critic, modulation_size,modulation_params_perc, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization
        
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env, 
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
            self.reward_components,
            [self.context_dim],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        max_torque = self.env.env.max_torque
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        #pi_curr = 0.05 # gradually update this 
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                max_torques = self.env.env.max_torque[:,:] # all joints are the same
                # modify actor with modulator network
                
                residual_actor = self.new_actor_critic.actor
                residual_critic = self.new_actor_critic.critic
                weight_to_add = {"actor":[], "critic":[]}
                bias_to_add = {"actor":[], "critic":[]}
                res_input_actor = max_torques.clone()#.reshape(-1,1)
                res_input_critic = max_torques.clone()

                #residual_layers_actor = [layer for layer in residual_actor.children() if isinstance(layer, nn.Linear)]
                #print([layer for layer in residual_actor.children()])
                #assert len(residual_layers_actor) > 0 # this will probably fail
                #res_input_actor = residual_layers_actor[0](res_input_orig)
                
                for idx,res_layer_actor in enumerate(residual_actor):
                    if idx==0 or not "Linear" in str(type(res_layer_actor)): # either activation or the linear layer
                        res_input_actor = res_layer_actor(res_input_actor)
                        continue
                    delta_actor = res_input_actor.unsqueeze(1) * res_layer_actor.weight.unsqueeze(0)
                    if len(weight_to_add["actor"]) < 3:
                        delta_actor = delta_actor[:, :math.ceil(delta_actor.shape[1] / self.modulation_size), :math.ceil(delta_actor.shape[2] / self.modulation_size)]
                    else:
                        delta_actor = delta_actor[:,:,:math.ceil(delta_actor.shape[2]/self.modulation_size)]
                    mod_param_limit = int(delta_actor.shape[-1] * self.modulation_params_perc)
                    delta_actor = delta_actor[:, :, :mod_param_limit]
                    #if it < 50:
                    #    delta_actor *= 0.0
                    weight_to_add["actor"].append(delta_actor)
                    if len(bias_to_add["actor"]) < 3:
                        bias = res_layer_actor.bias[:math.ceil(res_layer_actor.bias.shape[0]/self.modulation_size)]
                    else:
                        bias = res_layer_actor.bias
                    #if it < 50:
                    #    bias *= 0.0
                    mod_bias_limit = int(bias.shape[0] * self.modulation_params_perc)
                    bias = bias[:mod_bias_limit]
                    bias_to_add["actor"].append(bias)
                    res_input_actor = res_layer_actor(res_input_actor)

                for idx,res_layer_critic in enumerate(residual_critic):
                    if idx==0 or not "Linear" in str(type(res_layer_critic)): # either activation or the linear layer
                        res_input_critic = res_layer_critic(res_input_critic)
                        continue
                    delta_critic = res_input_critic.unsqueeze(1) * res_layer_critic.weight.unsqueeze(0)
                    if len(weight_to_add["critic"]) < 3:
                        delta_critic = delta_critic[:,:math.ceil(delta_critic.shape[1]/self.modulation_size),:math.ceil(delta_critic.shape[2]/self.modulation_size)]
                    else:
                        delta_critic = delta_critic[:,:,:math.ceil(delta_critic.shape[2]/self.modulation_size)]
                    mod_param_limit = int(delta_critic.shape[-1] * self.modulation_params_perc)
                    delta_critic = delta_critic[:, :, :mod_param_limit]
                    weight_to_add["critic"].append(delta_critic)
                    if len(bias_to_add["critic"]) < 3:
                        bias = res_layer_critic.bias[:math.ceil(res_layer_critic.bias.shape[0]/self.modulation_size)]
                    else:
                        bias = res_layer_critic.bias
                    mod_bias_limit = int(bias.shape[0] * self.modulation_params_perc)
                    bias = bias[:mod_bias_limit]
                    bias_to_add["critic"].append(bias)
                    res_input_critic = res_layer_critic(res_input_critic)

                #print([x.shape for x in weight_to_add["critic"]])
                #assert False
                max_torques = max_torques.clone()#.reshape((-1,1))
                for i in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs, critic_obs, self.base_ac, weight_to_add, bias_to_add, max_torques)
                    obs, rewards, dones, infos = self.env.step(actions)
                    obs = self.obs_normalizer(obs)
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    else:
                        critic_obs = obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    if self.reward_components == 1:
                        rewards = rewards.sum(dim=1, keepdim=True)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += torch.sum(rewards.sum(axis=1), dim=0)
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                #pi_curr = min(0.25, pi_curr + 0.0025) 
                self.alg.compute_returns(critic_obs,self.base_ac, weight_to_add, bias_to_add)
                self.env.env.update_curriculum(it)

            mean_value_loss, mean_surrogate_loss, mean_component_value_loss = self.alg.update(self.new_actor_critic, it)
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))
        #self.writer.add_scalar("Constraint satisfaction",torch.prod(1- torch.mean(locs["constraints"],dim=0)), locs["it"])
        #for i in range(len(locs["constraint_names"])):
        #    self.writer.add_scalar(f"Constraint satisfaction/{locs['constraint_names'][i]}", 1 - torch.mean(locs["constraints"][:, i]), locs["it"])
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        #for i in range(len(locs['mean_component_value_loss'])):
        #    self.writer.add_scalar(f"Loss/component_value_function_{self.env.env._episode_sums.keys()[i]}", locs['mean_component_value_loss'][i], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        #log_string += f"{'Constraint satisfaction:':>{pad}} {torch.prod(1 - torch.mean(locs['constraints'], dim=0)):.2f}\n"

        #for i in range(len(locs["constraint_names"])):
        #    constraint_name = locs["constraint_names"][i]
        #    constraint_value = 1 - torch.mean(locs["constraints"][:, i])
        #    log_string += f"{('Constraint_satisfaction_' + constraint_name):>{pad}} {constraint_value:.2f}\n"

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        # not implemented yet
        # deep copy actor
        #actor = deepcopy(self.base_actor)
        #residual_actor = self.new_actor_critic.actor
        #res_input = torch.tensor([1.0], device=self.device) 

        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

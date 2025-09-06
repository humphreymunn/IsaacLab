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
import math

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
        gradnorm=False,
        reward_component_names=None,
        reward_component_task_rew=None,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        if gradnorm:
            self.loss_weights = torch.nn.Parameter(
                torch.ones(self.actor_critic.num_reward_components, device=self.device),
                requires_grad=True
            )

            self.optimizer = optim.AdamW(
                list(self.actor_critic.parameters()) + [self.loss_weights],
                lr=learning_rate
            )
        else:
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
        self.gradnorm_alpha = 1.5  # usually between 1.0 and 2.0
        # self.loss_weights = torch.nn.Parameter(torch.ones(self.actor_critic.num_reward_components, device=self.device), requires_grad=True)
        self.initial_losses = None  # to be set after first iteration
        self.gradvac_beta = 1e-2           # EMA step (≈ 100-step window recommended)
        self.gradvac_phi = None            # will hold [C, C] EMA target matrix

        self.cagrad_c = 0.5          # trade-off knob (0 <= c < 1). 0.4–0.6 typical
        self.cagrad_steps = 25       # simplex PGD steps for the dual
        self.cagrad_lr = 0.1         # step size for the dual

        self.reward_component_names = reward_component_names
        self.reward_component_task_rew = reward_component_task_rew

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
        #reward_components = self.transition.rewards.shape[1]
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

    def compute_returns_multi(self, last_critic_obs):
        #self.ci_max = self.storage.adjust_rewards(self.ci_max, pi_curr)
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns_multi(last_values, self.gamma, self.lam)

    def _unflatten_to_param_shapes(self, flat_vec, like_grads):
        # like_grads: List[Tensor] from one objective to get shapes
        out, start = [], 0
        for g in like_grads:
            n = g.numel()
            out.append(flat_vec[start:start+n].view_as(g))
            start += n
        return out

    def _flatten_grads(self, grads_per_obj):
        # grads_per_obj: List[List[Tensor]]  -> [m, D]
        return torch.stack([torch.cat([g.view(-1) for g in grads]) for grads in grads_per_obj])

    def _adam_rms_flat(self, eps=1e-8):
        """
        Returns a flat vector r of per-parameter RMS = sqrt(v) + eps
        aligned with self.actor_critic.parameters() order.
        Falls back to ones if state not initialized yet.
        """
        parts = []
        for p in self.actor_critic.parameters():
            if not p.requires_grad:
                parts.append(torch.ones_like(p).flatten())
                continue
            st = self.optimizer.state.get(p, None)
            if st is None or ('exp_avg_sq' not in st):
                parts.append(torch.ones_like(p).flatten())
            else:
                v = st['exp_avg_sq']
                parts.append((v.sqrt() + eps).flatten())
        return torch.cat(parts)  # [D]

    @torch.no_grad()
    def apply_gradvac(self, objective_grads,
                    delta=0.05,      # gating margin
                    rho=0.3,         # softness toward EMA target
                    a2_clip=0.7,     # cap on mixing coeff
                    max_angle_deg=12 # cap on per-pair rotation
                    ):
        device = self.device
        m = len(objective_grads)
        if (self.gradvac_phi is None) or (self.gradvac_phi.shape[0] != m):
            self.gradvac_phi = torch.zeros((m, m), device=device)

        # Flatten grads [m, D]
        G  = self._flatten_grads(objective_grads)    # original per-component grads
        G0 = G.clone()                                # evolving left operand

        # Adam metric whitening vector r (sqrt(v)+eps), flat [D]
        r = self._adam_rms_flat(eps=1e-8)
        # Precompute whitened views
        G_tilde  = G  / r          # [m, D]
        G0_tilde = G0 / r          # [m, D]

        eps = 1e-12
        norms_G_t = G_tilde.norm(dim=1).clamp_min(eps)  # ||g_j||_M (fixed, original g_j)

        # For downscale-only guard (use Euclidean; we want to control actual param step)
        g_sum_orig = G.sum(dim=0)
        norm_sum_orig = g_sum_orig.norm().clamp_min(eps)

        for i in range(m):
            js = [j for j in range(m) if j != i]
            random.shuffle(js)
            for j in js:
                gi_t = G0_tilde[i]           # whitened evolving gi
                gj_t = G_tilde[j]            # whitened fixed gj
                ni_t = gi_t.norm().clamp_min(eps)
                nj_t = norms_G_t[j]

                # current cosine in Adam metric
                phi = (gi_t @ gj_t) / (ni_t * nj_t)
                phi = phi.clamp(-0.999999, 0.999999)

                # EMA target and softened target
                phi_hat = self.gradvac_phi[i, j].clamp(-0.999999, 0.999999)
                phi_tgt = phi + rho * (phi_hat - phi)

                # gate: only act if clearly below both EMA and a small negative threshold
                if (phi < phi_tgt) and (phi < -delta):
                    # closed-form a2 to reach phi_tgt (derived in Euclidean, valid in whitened space)
                    sqrt1_phi2  = torch.sqrt((1.0 - phi     * phi    ).clamp_min(1e-12))
                    sqrt1_pt2   = torch.sqrt((1.0 - phi_tgt * phi_tgt).clamp_min(1e-12))
                    a2 = (ni_t * (phi_tgt * sqrt1_phi2 - phi * sqrt1_pt2)) / (nj_t * sqrt1_pt2 + eps)

                    # cap the mix
                    a2 = a2.clamp(-a2_clip, a2_clip)

                    # propose update in both spaces (note: a2 is invariant under diagonal whitening)
                    gi_new      = G0[i]      + a2 * G[j]
                    gi_new_t    = G0_tilde[i] + a2 * G_tilde[j]

                    # cap max rotation (measured in Adam metric)
                    cos_rot = (G0_tilde[i] @ gi_new_t) / (G0_tilde[i].norm().clamp_min(eps) * gi_new_t.norm().clamp_min(eps))
                    cos_rot = cos_rot.clamp(-1.0, 1.0)
                    angle_deg = torch.acos(cos_rot) * (180.0 / math.pi)
                    if angle_deg > max_angle_deg:
                        a2 = a2 * (max_angle_deg / (angle_deg + 1e-6))
                        gi_new   = G0[i]       + a2 * G[j]
                        gi_new_t = G0_tilde[i] + a2 * G_tilde[j]

                    # commit
                    G0[i]       = gi_new
                    G0_tilde[i] = gi_new_t

                # EMA update with observed current cosine (in Adam metric)
                self.gradvac_phi[i, j] = (1.0 - self.gradvac_beta) * self.gradvac_phi[i, j] + self.gradvac_beta * phi

        # combine (sum semantics)
        combined_flat = G0.sum(dim=0)

        # downscale-only guard (Euclidean)
        norm_combined = combined_flat.norm().clamp_min(eps)
        scale = torch.minimum(torch.tensor(1.0, device=device), norm_sum_orig / norm_combined)
        combined_flat = combined_flat * scale

        return self._unflatten_to_param_shapes(combined_flat, objective_grads[0])

    def _proj_simplex(self, w):
        # Project onto probability simplex {w: sum w = 1, w >= 0}
        # Duchi et al. (2008)
        u, _ = torch.sort(w, descending=True)
        css = torch.cumsum(u, dim=0)
        rho = torch.nonzero(u > (css - 1) / torch.arange(1, w.numel()+1, device=w.device), as_tuple=False)[-1].item()
        theta = (css[rho] - 1.0) / (rho + 1.0)
        return torch.clamp(w - theta, min=0.0)

    @torch.no_grad()
    def apply_cagrad(self, objective_grads, c=0.5, steps=25, lr=0.1, eps=1e-12, downscale_only=True):
        """
        Conflict-Averse Gradient Descent (whole-model).
        objective_grads: List[List[param_tensor]] per objective (length m)
        c in [0,1): radius parameter from the paper (Alg.1); typical 0.2–0.6.
        """
        device = self.device
        m = len(objective_grads)

        # Flatten: G_rows = [m, D] (each row = g_i^T)
        G_rows = self._flatten_grads(objective_grads)                 # [m, D]
        g0 = G_rows.mean(dim=0)                                       # [D]
        phi = (c * g0.norm().clamp_min(eps)) ** 2                     # scalar

        # Gram matrix K = G G^T (m x m); and b = K 1
        K = G_rows @ G_rows.t()                                       # [m, m]
        ones = torch.ones(m, device=device)
        b = K @ ones                                                  # [m]

        # Solve min_{w in simplex} f(w) = (1/m) b^T w + sqrt(phi) * sqrt(w^T K w)
        w = torch.full((m,), 1.0 / m, device=device)
        for _ in range(steps):
            Kw = K @ w
            s = torch.dot(w, Kw).clamp_min(eps)                       # w^T K w
            grad = (b / m) + (math.sqrt(phi) * Kw) / torch.sqrt(s)    # ∇f(w)
            w = w - lr * grad
            w = self._proj_simplex(w)

        # Build gw = (1/m) sum_i w_i * g_i, then d* = g0 + sqrt(phi)/||gw|| * gw
        gw = (w @ G_rows) / m                                         # [D]
        gw_norm = gw.norm().clamp_min(eps)
        d = g0 + (math.sqrt(phi) / gw_norm) * gw                      # [D]

        # (Optional) downscale-only guard to avoid larger-than-baseline step in PPO
        if downscale_only:
            g_sum_orig = G_rows.sum(dim=0)
            scale = torch.minimum(torch.tensor(1.0, device=device),
                                g_sum_orig.norm().clamp_min(eps) / d.norm().clamp_min(eps))
            d = d * scale

        # Unflatten back to param-shaped tensors
        combined = []
        start = 0
        for g in objective_grads[0]:
            n = g.numel()
            combined.append(d[start:start+n].view_as(g))
            start += n
        return combined

    @torch.no_grad()
    def apply_pcgrad(self, objective_grads, normpres: bool = False, gradnorm: bool = False):
        num_objectives = len(objective_grads)
        eps = 1e-8

        # Flatten per-objective grads: [m, D]
        flat_grads = torch.stack([
            torch.cat([g.flatten() for g in grad_list])
            for grad_list in objective_grads
        ])  # [num_objectives, D]

        device = flat_grads.device
        projected_grads = flat_grads.clone()

        # Norm preservation baseline
        g_sum_orig = flat_grads.sum(dim=0)
        target_norm = g_sum_orig.norm().clamp_min(eps)

        # Task mask (aligned with reward order)
        is_task = torch.tensor(
            [name in self.reward_component_task_rew for name in self.reward_component_names],
            device=device, dtype=torch.bool
        )

        # Non-deterministic order for i
        perm_i = torch.randperm(num_objectives, device=device)
        for i in perm_i.tolist():
            gi = projected_grads[i]

            # Non-deterministic order for j each time
            perm_j = torch.randperm(num_objectives, device=device).tolist()

            if is_task[i]:
                # Task–task: project task i against other tasks
                for j in perm_j:
                    if j == i or not is_task[j]:
                        continue
                    gj = projected_grads[j]
                    dot = torch.dot(gi, gj)
                    nt2 = torch.dot(gj, gj)
                    if (dot < 0.0) and (nt2 > 1e-12):
                        gi = gi - (dot / (nt2 + eps)) * gj
                projected_grads[i] = gi
            else:
                # Penalty–task: project penalty i against tasks only
                for j in perm_j:
                    if not is_task[j]:
                        continue
                    gj = projected_grads[j]
                    dot = torch.dot(gi, gj)
                    nt2 = torch.dot(gj, gj)
                    if (dot < 0.0) and (nt2 > 1e-12):
                        gi = gi - (dot / (nt2 + eps)) * gj
                projected_grads[i] = gi

        # Combine grads (sum semantics)
        combined_flat = projected_grads.sum(dim=0)

        # Optional norm preservation
        if normpres:
            curr_norm = combined_flat.norm().clamp_min(eps)
            scale = torch.minimum(
                torch.tensor(1.0, device=combined_flat.device, dtype=combined_flat.dtype),
                target_norm / curr_norm
            )
            combined_flat = combined_flat * scale

        # Unflatten back to param shapes
        combined = []
        start = 0
        for g in objective_grads[0]:
            n = g.numel()
            combined.append(combined_flat[start:start + n].view_as(g))
            start += n
        return combined



    def update_multihead(self, pcgrad=False, gradnorm=False, normpres=False):
        mean_value_loss = 0
        mean_component_value_loss = torch.zeros((self.actor_critic.num_reward_components), device=self.device)
        mean_surrogate_loss = 0
        mean_entropy = 0.0
        mean_approx_kl = 0.0
        mean_clip_fraction = 0.0

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
            mean_entropy += entropy_batch.mean().item()

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
                    mean_approx_kl += kl.mean().item()
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            log_ratio = (actions_log_prob_batch - old_actions_log_prob_batch.squeeze()).clamp(-20, 20)
            ratio = log_ratio.exp().unsqueeze(1)
            #if ratio.dim() == 2 and ratio.size(1) == 1:
            #    ratio = ratio.squeeze(1)                     # [B]
            #else:
            #    ratio = ratio.view(-1)                       # [B]
            clip_fraction = ((ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param)).float().mean()
            mean_clip_fraction += clip_fraction.item()

            component_advantages_batch = component_advantages_batch  # shape: [B, C]
            #A_total = component_advantages_batch.sum(dim=1)  # [B]
            #eps = self.clip_param
            # r_eff = min(ratio, 1+eps) if A_total>=0 else max(ratio, 1-eps)
            #r_eff = torch.where(
            #    A_total >= 0,
            #    torch.minimum(ratio, torch.tensor(1.0 + eps, device=ratio.device)),
            #    torch.maximum(ratio, torch.tensor(1.0 - eps, device=ratio.device))
            #)                        

            # apply SAME effective ratio to all components
            #component_surrogate_losses = -(r_eff[:, None] * component_advantages_batch)   # [B, C]
            #mean_component_surrogate_loss = component_surrogate_losses.mean(dim=0)        # [C]

                        # [B, C]
            #mean_component_surrogate_loss = comp_losses.mean(dim=0)                     # [C]

            surrogate_per_component = -component_advantages_batch * ratio  # shape: [B, C]
            surrogate_per_component_clipped = -component_advantages_batch * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )

            component_surrogate_losses = torch.max(
                surrogate_per_component, surrogate_per_component_clipped
            )  # [B, C]

            mean_component_surrogate_loss = component_surrogate_losses.mean(dim=0)  # [C]
            # handles lstm case 
            #x = component_surrogate_losses
            #x = x if x.dim() == 3 else x.unsqueeze(0)  # [1, B, C] → [T, C, B] fake-T=1
            #x = x.permute(1, 0, 2).reshape(x.shape[1], -1)  # [C, T*B]
            #mean_component_surrogate_loss = x.mean(dim=1)  # [C]

            surrogate_loss = mean_component_surrogate_loss.sum()

            # Value function loss
            if self.use_clipped_value_loss:
                scale = returns_batch.std(dim=0,unbiased=False).clamp_min(1e-3)
                w_k   = (1.0 / (scale**2)).detach() # weight MSE by inverse of variance
                eps_k = (self.clip_param * scale).detach()

                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -eps_k, eps_k
                )
                value_losses = (value_batch - returns_batch).pow(2) #* w_k
                value_losses_clipped = (value_clipped - returns_batch).pow(2) #* w_k
                dim_use = 2 if len(value_losses.shape) > 2 else 1
                value_loss = torch.max(value_losses.sum(dim=dim_use), value_losses_clipped.sum(dim=dim_use)).mean()
                component_value_loss = torch.max(value_losses, value_losses_clipped).mean(dim=dim_use-1)  # shape: [C]
                if dim_use == 2:
                    mean_component_value_loss += component_value_loss.sum(dim=0).detach()
                else:
                    mean_component_value_loss += component_value_loss.detach()
            else:
                assert False # not implemented
                value_loss = (returns_batch - value_batch).pow(2).sum(dim=2).mean()
                mean_component_value_loss += (returns_batch - value_batch).pow(2).sum(dim=2).mean(dim=0).mean(dim=0)

            # More efficient per-component gradient computation
            self.optimizer.zero_grad()
            
            if self.initial_losses is None:
                self.initial_losses = mean_component_surrogate_loss.detach()

            self.current_losses = mean_component_surrogate_loss.detach()

            # Compute gradients for all components in one backward pass
            # Create a tensor of ones for each component to compute gradients
            if not pcgrad:
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                loss.backward()
            else: # pcgrad
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
                projected_grads = self.apply_pcgrad(component_grads, normpres, gradnorm)

                # Assign PCGrad-updated gradients to model
                for param, g in zip(self.actor_critic.parameters(), projected_grads):
                    if param.requires_grad:
                        param.grad = g

                # Value loss + entropy (backward separately and add)
                value_loss.backward(retain_graph=True)
                if self.entropy_coef > 0:
                    (-self.entropy_coef * entropy_batch.mean()).backward()

            # don't clip discriminator gradients
            actor_critic_params = list(self.actor_critic.actor.parameters()) + list(self.actor_critic.critic.parameters())
            nn.utils.clip_grad_norm_(actor_critic_params, self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_entropy /= num_updates
        mean_approx_kl /= num_updates
        mean_clip_fraction /= num_updates
        mean_value_loss /= num_updates
        mean_component_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        def _flatten_grads(grads_list):
            return torch.stack([torch.cat([g.view(-1) for g in grads])
                                for grads in grads_list])  # [C, D]

        grad_overall_conflict_pct = None
        if pcgrad:
            G = _flatten_grads(component_grads).detach()          # [C, D]
            # Pairwise dot matrix
            D = G @ G.t()                                         # [C, C]
            # Only count each pair once (upper triangle, no diagonal)
            C = G.size(0)
            iu = torch.triu_indices(C, C, offset=1, device=G.device)
            # Angle > 90°  <=> dot < 0  (no need to compute acos)
            conflicts = (D[iu[0], iu[1]] < 0.0).float()           # [P], P=C*(C-1)/2
            grad_overall_conflict_pct = (conflicts.mean() * 100.0).item()

        def compute_grad_angles(grads_list):
            """Returns (angles, labels, pairs).
            angles: list[float] (degrees)
            labels: list[str]  e.g., 'rewardA_vs_rewardB'
            pairs:  list[tuple[int,int]] (i,j) indices
            """
            flat_grads = [torch.cat([g.view(-1) for g in grads]) for grads in grads_list]
            angles, labels, pairs = [], [], []

            # Prefer names from PPO; fallback to generic
            try:
                names = list(self.reward_component_names)
            except Exception:
                names = [f"C{i}" for i in range(len(flat_grads))]

            for i in range(len(flat_grads)):
                for j in range(i + 1, len(flat_grads)):
                    cos_sim = torch.nn.functional.cosine_similarity(flat_grads[i], flat_grads[j], dim=0).clamp(-1, 1)
                    angle = torch.acos(cos_sim) * (180 / torch.pi)
                    angles.append(angle.item())
                    labels.append(f"{names[i]}_vs_{names[j]}")
                    pairs.append((i, j))
            return angles, labels, pairs

        grad_angles, grad_angle_labels, grad_angle_pairs = None, None, None
        if pcgrad:
            grad_angles, grad_angle_labels, grad_angle_pairs = compute_grad_angles(component_grads)

        return {
            "mean_value_loss": mean_value_loss,
            "mean_surrogate_loss": mean_surrogate_loss,
            "mean_component_value_loss": mean_component_value_loss,
            "mean_component_surrogate_loss": mean_component_surrogate_loss,
            "mean_entropy": entropy_batch.mean().item(),
            "mean_clip_fraction": (torch.logical_or(ratio < 1.0 - self.clip_param, ratio > 1.0 + self.clip_param).float().mean().item()),
            "mean_approx_kl": kl_mean.item() if self.desired_kl is not None else 0.0,
            "per_head_advantages": component_advantages_batch.mean(dim=0).detach().cpu(),
            "grad_norms": [
                torch.norm(torch.stack([torch.norm(g.detach()) for g in grads if g is not None]))
                for grads in component_grads
            ] if pcgrad else None,
            "grad_angles": grad_angles,
            "grad_overall_conflict_pct": grad_overall_conflict_pct,  # None if not pcgrad/gradvac
            "grad_angle_labels": grad_angle_labels,   # <— NEW
            "grad_angle_pairs": grad_angle_pairs,     # <— optional (indices)
            "grad_projection_magnitude": (sum((torch.norm(g.detach()) for g in projected_grads)) / len(projected_grads)) if pcgrad else None,
            "action_magnitudes": actions_batch.abs().mean(dim=0).detach().cpu(),
        }
    
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0.0
        mean_approx_kl = 0.0
        mean_clip_fraction = 0.0

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
            mean_entropy += entropy_batch.mean().item()

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
                    mean_approx_kl += kl.mean().item()
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            clip_fraction = ((ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param)).float().mean()
            mean_clip_fraction += clip_fraction.item()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                dim_use = 2 if len(value_losses.shape) > 2 else 1
                value_loss = torch.max(value_losses.sum(dim=dim_use), value_losses_clipped.sum(dim=dim_use)).mean()
                #component_value_loss = torch.max(value_losses, value_losses_clipped).mean(dim=0)  # shape: [C]
                #mean_component_value_loss += component_value_loss.detach()

            else:
                assert False # not implemented
                value_loss = (returns_batch - value_batch).pow(2).sum(dim=2).mean()
                mean_component_value_loss += (returns_batch - value_batch).pow(2).sum(dim=2).mean(dim=0).mean(dim=0)

            # More efficient per-component gradient computation
            
            
            # Compute gradients for all components in one backward pass
            # Create a tensor of ones for each component to compute gradients
            
            
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            self.optimizer.zero_grad()
            loss.backward()

            actor_critic_params = list(self.actor_critic.actor.parameters()) + list(self.actor_critic.critic.parameters())
            nn.utils.clip_grad_norm_(actor_critic_params, self.max_grad_norm)
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
        mean_entropy /= num_updates
        mean_approx_kl /= num_updates
        mean_clip_fraction /= num_updates
        mean_value_loss /= num_updates
        #mean_component_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return {
            "mean_value_loss": mean_value_loss,
            "mean_surrogate_loss": mean_surrogate_loss,
            "mean_component_value_loss": None,
            "mean_component_surrogate_loss": None,
            "mean_entropy": entropy_batch.mean().item(),
            "mean_clip_fraction": (torch.logical_or(ratio < 1.0 - self.clip_param, ratio > 1.0 + self.clip_param).float().mean().item()),
            "mean_approx_kl": kl_mean.item() if self.desired_kl is not None else 0.0,
            "per_head_advantages": None,
            "grad_norms": None,
            "grad_projection_magnitude": None,
            "action_magnitudes": actions_batch.abs().mean(dim=0).detach().cpu(),
        }


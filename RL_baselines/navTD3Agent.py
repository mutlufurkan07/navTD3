import os
import copy
import random
from math import pi, cos
import time
import numpy as np

from network import ActorNetwork, CriticNetwork
from memory import Memory
import torch
import torch.optim as optim
import torch.nn as nn


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


class NAVTD3Agent:
    def __init__(self,
                 device,
                 mem_size,
                 batch_size,
                 state_dim,
                 gamma,
                 tau,
                 actor_lr,
                 critic_lr,
                 std,
                 action_dim,
                 noise_mag,
                 max_action,
                 smoothing_noise_limit,
                 save_path,
                 linear_noise_decay=25000):
        self.save_path = save_path
        self.device = device
        self.memory = Memory(mem_size=mem_size,
                             batch_size=batch_size,
                             state_dim=state_dim,
                             action_dim=action_dim,
                             device=self.device)
        self.linear_noise_decay = linear_noise_decay
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.curr_reward = 0
        self.noise_mag = noise_mag
        self.max_action = max_action
        self.noise_limit = smoothing_noise_limit

        self.actor = ActorNetwork(state_dims=state_dim,
                                  layer1_dims=256, layer2_dims=256,
                                  action_dim=action_dim, mode="NAVTD3", max_action=max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic1 = CriticNetwork(state_dim=state_dim,
                                     action_dims=action_dim,
                                     layer1_dims=256,
                                     layer2_dims=256)
        self.critic1_target = copy.deepcopy(self.critic1)

        self.critic2 = CriticNetwork(state_dim=state_dim,
                                     action_dims=action_dim,
                                     layer1_dims=256,
                                     layer2_dims=256)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=actor_lr)
        self.critic1_optim = optim.Adam(params=self.critic1.parameters(), lr=critic_lr)
        self.critic2_optim = optim.Adam(params=self.critic2.parameters(), lr=critic_lr)

        self.critic_criterion = nn.MSELoss()

        self.gamma = gamma
        self.tau = tau
        self.std = std
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic1.to(self.device)
        self.critic2.to(self.device)
        self.critic1_target.to(self.device)
        self.critic2_target.to(self.device)

    def add_noise(self, t):
        noise = self.noise_mag * np.random.normal(0, self.std, self.action_dim)
        noise = torch.tensor(noise).unsqueeze(0)
        steering_cos_noise_period = 50  # random.uniform(44, 55)  # 50
        w = 0.49 * pi / steering_cos_noise_period
        w2 = 0.98 * pi / steering_cos_noise_period
        steering_cos_noise = cos(w * t) * 1.2
        acc_cos_noise = (cos(w2 * t) / 1.3) + 0.4
        mkk = -(1 * t) / self.linear_noise_decay + 1
        mkk = max(0.1, mkk)
        mkk2 = max(0.4, mkk)
        steering_cos_noise = steering_cos_noise * mkk
        acc_cos_noise = acc_cos_noise * mkk2
        if t < 10000:
            noise[0, 1] += steering_cos_noise
            noise[0, 0] += acc_cos_noise
        return noise

    def action_selection(self, state, t, eval=False):
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action = self.actor.forward(state)[0].cpu()
        if not eval:
            action = (action + self.add_noise(t)).squeeze(0)
        return action.numpy().clip(-self.max_action, self.max_action)

    def update(self, is_policy_update):
        if self.memory.mem_index <= self.batch_size:
            return

        if self.device != torch.device("cuda"):
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, gradmult_batch, batch_index = self.memory.move2cuda(self.memory.sample(), device=self.device)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, gradmult_batch, batch_index = self.memory.sample()

        noise = (torch.randn_like(action_batch) * self.noise_mag).to(self.device).clamp(-self.noise_limit, self.noise_limit)
        next_actions = (self.actor_target.forward(next_state_batch)[0] + noise).clamp(-self.max_action, self.max_action)

        action_batch = action_batch.clamp(-self.max_action, self.max_action)
        Q1_pred = self.critic1.forward(state_batch, action_batch)
        Q2_pred = self.critic2.forward(state_batch, action_batch)

        Q1 = self.critic1_target(next_state_batch, next_actions)
        Q2 = self.critic2_target(next_state_batch, next_actions)
        y = torch.min(Q1, Q2)

        with torch.no_grad():
            y = reward_batch.unsqueeze(1) + (self.gamma * y * (1 - terminal_batch.unsqueeze(1)))

        critic1_loss = self.critic_criterion(Q1_pred, y)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = self.critic_criterion(Q2_pred, y)
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        if is_policy_update:
            # Actor loss
            actor_forward, before_tanh = self.actor.forward(state_batch)
            before_tanh[before_tanh == before_tanh.clip(-0.9, 0.9)] = 0

            policy_loss = (-self.critic1.forward(state_batch, actor_forward) + before_tanh ** 2).mean()

            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

        # soft update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_models(self):
        a1 = os.path.join(self.save_path, "actor.pth")
        c1 = os.path.join(self.save_path, "critic.pth")
        torch.save(self.actor.state_dict(), a1)
        torch.save(self.critic1.state_dict(), c1)

    def load_models(self, episode, exp_id):
        self.actor.load_state_dict(
            torch.load(f"model_params_td3/{exp_id}_{episode}_actor.pth", map_location=self.device))
        self.actor_target.load_state_dict(
            torch.load(f"model_params_td3/{exp_id}_{episode}_actortarget.pth", map_location=self.device))
        self.critic1.load_state_dict(
            torch.load(f"model_params_td3/{exp_id}_{episode}_critic.pth", map_location=self.device))
        self.critic1_target.load_state_dict(
            torch.load(f"model_params_td3/{exp_id}_{episode}_critictarget.pth", map_location=self.device))

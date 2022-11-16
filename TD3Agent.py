import os
import random

from network import ActorNetwork, CriticNetwork
from memory import Memory
import numpy as np
import torch.optim as optim
import copy
import torch
import torch.nn as nn
import time
from math import pi, cos


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.autograd.profiler.emit_nvtx(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.set_detect_anomaly(False)


current_seed = 12
print(f"Seed is set to : {current_seed}")
set_seed(current_seed)


class TD3Agent:
    def __init__(self,
                 training_params,
                 save_path,
                 std,
                 action_dim,
                 cam_dim,
                 sketch_dim,
                 pos_dim,
                 noise_mag,
                 max_action,
                 smoothing_noise_limit,
                 linear_noise_decay=20000):
        self.save_path = save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = Memory(mem_size=training_params["experience_replay_size"],
                             batch_size=training_params["training_batch_size"], cam_state_dim=cam_dim,
                             sketch_dim=sketch_dim,
                             pos_state_dim=pos_dim, action_dim=action_dim,
                             steering_queue_state_dim=training_params["last_steering_queue_len"],
                             device=self.device)
        self.linear_noise_decay = linear_noise_decay
        state_dim = 7
        self.action_dim = action_dim
        self.gamma = training_params["gamma"]
        self.tau = training_params["tau"]
        self.batch_size = training_params["training_batch_size"]
        self.curr_reward = 0
        self.noise_mag = noise_mag
        self.max_action = max_action
        self.noise_limit = smoothing_noise_limit

        self.actor = ActorNetwork(frame_number=training_params["framestack_len"],
                                  camera_dim=cam_dim,
                                  state_dims=state_dim,
                                  past_steering_dims=training_params["last_steering_queue_len"],
                                  layer1_dims=training_params["actor_fc1_dim"],
                                  layer2_dims=training_params["actor_fc2_dim"],
                                  action_dim=action_dim)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic1 = CriticNetwork(frame_dim=training_params["framestack_len"],
                                     camera_dim=cam_dim,
                                     state_dim=state_dim,
                                     past_steering_dims=training_params["last_steering_queue_len"],
                                     action_dims=action_dim,
                                     layer1_dims=training_params["critic_fc1_dim"],
                                     layer2_dims=training_params["critic_fc2_dim"])
        self.critic1_target = copy.deepcopy(self.critic1)

        self.critic2 = CriticNetwork(frame_dim=training_params["framestack_len"],
                                     camera_dim=cam_dim,
                                     state_dim=state_dim,
                                     past_steering_dims=training_params["last_steering_queue_len"],
                                     action_dims=action_dim,
                                     layer1_dims=training_params["critic_fc1_dim"],
                                     layer2_dims=training_params["critic_fc2_dim"])
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=training_params["actor_learning_rate"])
        self.critic1_optim = optim.Adam(params=self.critic1.parameters(), lr=training_params["critic_learning_rate"])
        self.critic2_optim = optim.Adam(params=self.critic2.parameters(), lr=training_params["critic_learning_rate"])

        self.critic_criterion = nn.MSELoss()

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

    def action_selection(self, sketch, depth, pos, past_steerings, t):
        depth = torch.from_numpy(depth).to(self.device).unsqueeze(0)
        sketch = torch.from_numpy(sketch).to(self.device).unsqueeze(0)
        pos = torch.from_numpy(pos).to(self.device).unsqueeze(0)
        past_steerings = torch.from_numpy(past_steerings).to(self.device).unsqueeze(0)
        action = self.actor.forward(sketch, depth, pos, past_steerings)[0]
        acct = action.detach()
        acct = acct.cpu()
        act = (acct + self.add_noise(t)).squeeze(0)
        return act.numpy().clip(-self.max_action, self.max_action)

    def update(self, is_policy_update):
        if self.memory.mem_index <= self.batch_size:
            return

        # print("UP DATEEEEEEEEEEEE")

        sketch_batch, new_sketch_batch, depth_batch, new_depth_batch, pos_batch, new_pos_batch, past_action_batch, new_past_action_batch, action_batch, reward_batch, terminal_batch, gradmult_batch, batch_index = self.memory.sample()

        noise = (torch.randn_like(action_batch) * self.noise_mag).to(self.device).clamp(-self.noise_limit,
                                                                                        self.noise_limit)
        next_actions = (self.actor_target.forward(new_sketch_batch, new_depth_batch, new_pos_batch,
                                                  new_past_action_batch)[0] + noise).clamp(-self.max_action,
                                                                                           self.max_action)

        action_batch = action_batch.clamp(-self.max_action, self.max_action)
        Q1_pred = self.critic1.forward(sketch_batch, depth_batch, pos_batch, action_batch, past_action_batch)
        Q2_pred = self.critic2.forward(sketch_batch, depth_batch, pos_batch, action_batch, past_action_batch)

        Q1 = self.critic1_target(new_sketch_batch, new_depth_batch, new_pos_batch, next_actions, new_past_action_batch)
        Q2 = self.critic2_target(new_sketch_batch, new_depth_batch, new_pos_batch, next_actions, new_past_action_batch)
        y = torch.min(Q1, Q2)

        with torch.no_grad():
            y = reward_batch.unsqueeze(1) + (self.gamma * y * (1 - terminal_batch.unsqueeze(1)))

        critic1_loss = self.critic_criterion(Q1_pred, y)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1, norm_type=2)
        self.critic1_optim.step()

        critic2_loss = self.critic_criterion(Q2_pred, y)
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1, norm_type=2)
        self.critic2_optim.step()
        if is_policy_update:
            # Actor loss
            actor_forward, before_tanh = self.actor.forward(sketch_batch, depth_batch, pos_batch, past_action_batch)
            before_tanh[before_tanh == before_tanh.clip(-0.9, 0.9)] = 0

            policy_loss = (-self.critic1.forward(sketch_batch, depth_batch, pos_batch, actor_forward, past_action_batch)
                           + before_tanh ** 2).mean()

            # policy_loss = policy_loss * gradmult_batch.unsqueeze(1)
            # self.memory.gradmult_memory[batch_index] = 1.0

            # policy_loss = policy_loss.mean()

            self.actor_optim.zero_grad()
            policy_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1, norm_type=2)
            self.actor_optim.step()
        # print(time.time() - strt)

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
        #torch.save(self.actor_target.state_dict(), f"model_params_td3/{exp_id}_{episode}_actortarget.pth")
        torch.save(self.critic1.state_dict(), c1)
        #torch.save(self.critic1_target.state_dict(), f"model_params_td3/{exp_id}_{episode}_critictarget.pth")

    def load_models(self, episode, exp_id):

        self.actor.load_state_dict(
            torch.load(f"model_params_td3/{exp_id}_{episode}_actor.pth", map_location=self.device))
        self.actor_target.load_state_dict(
            torch.load(f"model_params_td3/{exp_id}_{episode}_actortarget.pth", map_location=self.device))
        self.critic1.load_state_dict(
            torch.load(f"model_params_td3/{exp_id}_{episode}_critic.pth", map_location=self.device))
        self.critic1_target.load_state_dict(
            torch.load(f"model_params_td3/{exp_id}_{episode}_critictarget.pth", map_location=self.device))

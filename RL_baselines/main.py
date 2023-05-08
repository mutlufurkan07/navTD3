import argparse
import os

import numpy as np
import pandas as pd
import torch

import gym
from navTD3Agent import NAVTD3Agent
from utils import evaluate_agent
from tqdm import tqdm

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLanderContinuous-v2", type=str)
    parser.add_argument("--MAXTIMESTAMP", default=1e6, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--trainingstart", default=25000, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")

    if os.path.exists("models") is False:
        os.mkdir("models")

    current_seed = args.seed
    set_seed(current_seed)
    print(f"Seed is set to : {current_seed}")

    env = gym.make(args.env)
    print(f"gym env is set to : {args.env}")
    evaluations = []

    agent = NAVTD3Agent(device=device,
                        mem_size=int(1e6),
                        batch_size=256,
                        state_dim=env.observation_space.shape[0],
                        gamma=0.99,
                        tau=0.005,
                        actor_lr=3e-4,
                        critic_lr=3e-4,
                        std=0.1,
                        action_dim=env.action_space.shape[0],
                        noise_mag=0.2,
                        max_action=float(env.action_space.high[0]),
                        smoothing_noise_limit=0.5,
                        save_path="models",
                        linear_noise_decay=10000)

    state, done = env.reset()[0], False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    for t in tqdm(range(int(args.MAXTIMESTAMP))):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.trainingstart:
            action = env.action_space.sample()
        else:
            action = agent.action_selection(state, t)

        next_state, reward, done, truncated, _ = env.step(action)
        done_bool = float(done or truncated) if episode_timesteps < env._max_episode_steps else 0

        agent.memory.store(state, action, reward, next_state, done_bool)
        state = next_state
        episode_reward += reward

        if t >= args.trainingstart:
            agent.update(is_policy_update=t % 2 == 0)
        if done or truncated:
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, done = env.reset()[0], False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t+1) % 3000 == 0:
            # Evaluate episode
            evaluations.append(evaluate_agent(agent, env_name=args.env, seed=current_seed, eval_episodes=5))

    pd.to_pickle(evaluations, f"evaluations_{args.env}_seed{current_seed}.pkl")
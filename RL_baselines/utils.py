import gym
import numpy as np


def evaluate_agent(agent, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    # eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(seed=seed + _)[0], False
        truncated = False
        while not (done or truncated):
            action = agent.action_selection(np.asarray(state, dtype=np.float32), t=-1, eval=True)
            state, reward, done, truncated, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward

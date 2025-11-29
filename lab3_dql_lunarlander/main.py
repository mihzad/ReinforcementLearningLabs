import gymnasium as gym
import torch
import numpy as np
from collections import deque
from agent import Agent


def train_dqn(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    print("Training started...")
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")

        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        # 200+ score => lands successfully
        if np.mean(scores_window) >= 200.0:
            print(f'\nEnvironment solved in {i_episode - 100:d} episodes.\tAvg Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_best.pth')
            break


def test(env, agent, num_episodes=5):
    print("\nStarting Test (Exploitation Mode)...")
    for i in range(num_episodes):
        state, _ = env.reset()
        score = 0
        while True:
            # act with eps=0.0 for pure exploitation
            action = agent.act(state, eps=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            state = next_state
            if terminated or truncated:
                break
        print(f"Test Episode {i + 1}: Score {score}")


if __name__ == '__main__':
    TRAIN_MODE = True

    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    if TRAIN_MODE:
        train_dqn(env, agent)
        env.close()
    else:
        try:
            agent.qnetwork_local.load_state_dict(torch.load('checkpoint_best.pth'))
            print("Loaded checkpoint successfully.")
        except FileNotFoundError:
            print("No checkpoint found! Please train first.")
            exit()

    print("Visualizing result...")
    env_viz = gym.make('LunarLander-v3', render_mode="human")

    test(env_viz, agent)
    env_viz.close()
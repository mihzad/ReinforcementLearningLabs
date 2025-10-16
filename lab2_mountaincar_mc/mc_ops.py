import numpy as np
import gymnasium as gym
from collections import defaultdict


class MountainCarMonteCarlo:
    def __init__(self, train_env, test_env, position_bins_count, velocity_bins_count, gamma=0.99, epsilon=0.1):
        self.pbc = position_bins_count
        self.vbc = velocity_bins_count
        self.env = train_env
        self.test_env = test_env
        self.gamma = gamma
        self.epsilon = epsilon

        self.n_actions = self.env.action_space.n
        self.Q = np.zeros((self.pbc, self.vbc, self.n_actions))
        self.returns = defaultdict(list)


    def discretize_state(self, state):
        pos_low, pos_high = self.env.observation_space.low[0], self.env.observation_space.high[0]
        pos_bin = self.convert_continuous_discrete(state[0], pos_low, pos_high, self.pbc)

        vel_low, vel_high = self.env.observation_space.low[1], self.env.observation_space.high[1]
        vel_bin = self.convert_continuous_discrete(state[1], vel_low, vel_high, self.vbc)

        return pos_bin, vel_bin

    def undiscretize_state(self, state):
        pos_low, pos_high = self.env.observation_space.low[0], self.env.observation_space.high[0]
        pos_restored = self.convert_discrete_continuous(state[0], pos_low, pos_high, self.pbc)

        vel_low, vel_high = self.env.observation_space.low[1], self.env.observation_space.high[1]
        vel_restored = self.convert_discrete_continuous(state[1], vel_low, vel_high, self.vbc)

        return pos_restored, vel_restored

    def convert_continuous_discrete(self, value, low, high, bins_count):
        if bins_count == 1:
            return 0
        bin_step = (high - low) / (bins_count - 1)
        bin_idx = int(round((value - low) / bin_step))
        return int(np.clip(bin_idx, 0, bins_count - 1))

    def convert_discrete_continuous(self, bin_idx, low, high, bins_count):
        if bins_count == 1:
            return (low + high) / 2
        return low + bin_idx * (high - low) / (bins_count - 1)


    def reward(self, old_pos, new_pos):
        return -1 + np.abs(new_pos - old_pos)
    
    def predict_next_state(self, curr_state, action):
        g = self.env.unwrapped.gravity
        force = self.env.unwrapped.force

        pos_low, pos_high = self.env.observation_space.low[0], self.env.observation_space.high[0]
        vel_low, vel_high = self.env.observation_space.low[1], self.env.observation_space.high[1]

        pos, vel = curr_state

        vel_next = vel + (action - 1) * force - g * np.cos(3 * pos)
        vel_next = np.clip(vel_next, vel_low, vel_high)

        pos_next = pos + vel_next
        pos_next = np.clip(pos_next, pos_low, pos_high)

        return pos_next, vel_next


    def epsilon_greedy(self, state):
        pos_bin, vel_bin = self.discretize_state(state)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[pos_bin, vel_bin])

    def generate_episode(self):
        """Run a full episode following the current ε-greedy policy."""
        episode = []
        state, _ = self.env.reset()
        done = False

        timer = 0
        while not done:
            action = self.epsilon_greedy(state)
            next_state = self.predict_next_state(state, action)
            episode.append((state, action, self.reward(old_pos=state[0], new_pos=next_state[0])))
            done = (next_state[0] >= self.env.unwrapped.goal_position) or timer >= 1000
            state = next_state
            timer += 1

        return episode

    def update_from_episode(self, episode):
        """First-visit Monte Carlo Q update."""

        first_occurrence = {}
        for t, (state, action, _) in enumerate(episode):
            pos_bin, vel_bin = self.discretize_state(state)
            key = (pos_bin, vel_bin, action)
            if key not in first_occurrence:
                first_occurrence[key] = t

        G = 0.0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            pos_bin, vel_bin = self.discretize_state(state)
            key = (pos_bin, vel_bin, action)

            G = self.gamma * G + reward

            # Update only if this t is the "first occurrence" of the key
            if first_occurrence[key] == t:
                self.returns[key].append(G)
                self.Q[pos_bin, vel_bin, action] = np.mean(self.returns[key])

    def derive_policy(self):
        """Return greedy policy based on learned Q."""
        return np.argmax(self.Q, axis=2)

    def train_mc(self, episodes=5000, verbose=True):
        """Main training loop for First-Visit Monte Carlo."""
        rewards = []

        for ep in range(1, episodes + 1):
            episode = self.generate_episode()
            self.update_from_episode(episode)
            total_reward = sum(r for (_, _, r) in episode)
            rewards.append(total_reward)

            if verbose and ep % 500 == 0:
                avg_r = np.mean(rewards[-500:])
                print(f"Episode {ep:5d}: avg reward {avg_r:.2f}")

        return rewards
    
    def test_policy(self, policy, max_steps=1000, render=True):
        state, _ = self.test_env.reset()
        if render:
            self.test_env.render()

        total_reward = 0
        for _ in range(max_steps):
            pos_bin, vel_bin = self.discretize_state(state=state)
            action = int(policy[pos_bin, vel_bin])
            state, reward, terminated, truncated, info = self.test_env.step(action)
            total_reward += reward

            if render:
                self.test_env.render()
            if terminated:
                break

        return total_reward

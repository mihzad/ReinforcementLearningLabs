import gymnasium as gym
import numpy as np
from tqdm import tqdm

from basic_ops import MountainCarIteration


class PolicyIteration(MountainCarIteration):
    def __init__(self, gym_env, position_bins_count, velocity_bins_count, gamma, reward_power=1):
        super().__init__(gym_env, position_bins_count, velocity_bins_count)
        self.gamma = gamma
        self.reward_power = reward_power

    def reward(self, old_pos, new_pos):
        return -1 + np.abs(new_pos - old_pos) ** self.reward_power


    def evaluate_value_function(self, policy, max_iter_count=1000, eps=10e-6):
        value_table = np.zeros((self.pbc, self.vbc))

        for _ in range(max_iter_count):
            previous_value_table = np.copy(value_table)

            for pos_bin_idx in range(self.pbc):
                for vel_bin_idx in range(self.vbc):
                    pos, vel = self.undiscretize_state(state=(pos_bin_idx, vel_bin_idx))
                    a = policy[pos_bin_idx, vel_bin_idx]

                    next_pos, next_vel = self.predict_next_state(curr_state=(pos, vel), action=a)
                    next_pos_bin, next_vel_bin = self.discretize_state(state=(next_pos, next_vel))

                    reward = self.reward(old_pos=pos, new_pos=next_pos)
                    value_table[pos_bin_idx, vel_bin_idx] = reward + self.gamma * previous_value_table[next_pos_bin, next_vel_bin]

            if np.sum(np.fabs(previous_value_table - value_table)) <= eps:
                break

        return value_table

    def extract_policy(self, value_table, gamma=0.99):
        policy = np.zeros((self.pbc, self.vbc))

        for pos_bin_idx in range(self.pbc):
            for vel_bin_idx in range(self.vbc):
                pos, vel = self.undiscretize_state(state=(pos_bin_idx, vel_bin_idx))

                q_values = []

                for a in range(self.env.action_space.n):
                    next_pos, next_vel = self.predict_next_state(curr_state=(pos, vel), action=a)
                    next_pos_bin, next_vel_bin = self.discretize_state(state=(next_pos, next_vel))
                    reward = -1 + np.abs(next_pos - pos)
                    q_value = reward + gamma * value_table[next_pos_bin, next_vel_bin]
                    q_values.append(q_value)

                policy[pos_bin_idx, vel_bin_idx] = np.argmax(q_values)

        return policy


    def perform(self, max_policy_updates=50, max_value_updates_per_policy=50, eps=10e-3):
        value_table = np.zeros((self.pbc, self.vbc))
        policy = np.zeros((self.pbc, self.vbc))

        for _ in tqdm(range(max_policy_updates)):
            old_policy = np.copy(policy)
            value_table = self.evaluate_value_function(
                old_policy, max_iter_count=max_value_updates_per_policy, eps=eps)
            policy = self.extract_policy(value_table)

            if np.array_equal(old_policy, policy):
                break

        return value_table, policy



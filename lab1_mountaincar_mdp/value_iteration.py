import gymnasium as gym
import numpy as np
import tqdm

from basic_ops import MountainCarIteration


class ValueIteration(MountainCarIteration):

    def __init__(self, gym_env, position_bins_count, velocity_bins_count, gamma):
        super().__init__(gym_env, position_bins_count, velocity_bins_count)
        self.gamma = gamma

    def perform(self, iter_count=1000, eps=1e-3):
        value_table = np.zeros((self.pbc, self.vbc))
        policy = np.zeros((self.pbc, self.vbc))

        for _ in tqdm.tqdm(range(iter_count)):
            previous_value_table = np.copy(value_table)

            for pos_bin_idx in range(self.pbc):
                for vel_bin_idx in range(self.vbc):
                    pos, vel = self.undiscretize_state(state=(pos_bin_idx, vel_bin_idx))
                    q_values = []

                    for a in range(self.env.action_space.n):
                        next_pos, next_vel = self.predict_next_state(curr_state=(pos, vel), action=a)
                        next_pos_bin, next_vel_bin = self.discretize_state(state=(next_pos, next_vel))

                        reward = -1 + np.abs(next_pos - pos)
                        q_value = reward + self.gamma * previous_value_table[next_pos_bin, next_vel_bin]
                        q_values.append(q_value)

                    value_table[pos_bin_idx, vel_bin_idx] = max(q_values)
                    policy[pos_bin_idx, vel_bin_idx] = np.argmax(q_values)

            print(f"eps = {np.sum(np.fabs(previous_value_table - value_table))}")
            if np.sum(np.fabs(previous_value_table - value_table)) <= eps:
                break

        return value_table, policy
from policy_iteration import PolicyIteration
from value_iteration import ValueIteration
from basic_ops import sanity_check
import gymnasium as gym
import numpy as np
import torch

if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="human")
    #sanity_check(env)

    #pbc=18 means that pos_bin_size = 0.1 and vbc=14 means vel_bin_size = 0.01. So i use em as a basis
    position_bins_count = 18*4
    velocity_bins_count = 14*4

    #policy_iter = PolicyIteration(env, position_bins_count, velocity_bins_count, gamma=0.8)
    #value_table, optimal_policy = policy_iter.perform(max_policy_updates=50, max_value_updates_per_policy=50, eps=10e-6)

    value_iter = ValueIteration(env, position_bins_count, velocity_bins_count, gamma=0.8)
    value_table, optimal_policy = value_iter.perform(iter_count=1000, eps=1e-6)

    #torch.save({"policy": optimal_policy}, "policyiter_optimal_policy.pth")
    torch.save({"policy": optimal_policy}, "valueiter_optimal_policy.pth")

    #policy_obj = torch.load("policyiter_optimal_policy.pth", weights_only=False)
    #policy_obj = torch.load("valueiter_optimal_policy.pth", weights_only=False)
    #optimal_policy = policy_obj["policy"]

    #total_reward = policy_iter.test_policy(optimal_policy)
    total_reward = value_iter.test_policy(optimal_policy)
    print(f"Total reward for value iteration policy: {total_reward}")

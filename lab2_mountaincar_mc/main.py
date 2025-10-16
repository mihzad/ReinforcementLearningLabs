from mc_ops import MountainCarMonteCarlo
import gymnasium as gym
import numpy as np
import torch
import time


if __name__ == "__main__":
    train_env = gym.make("MountainCar-v0", render_mode=None)
    test_env = gym.make("MountainCar-v0", render_mode="human")
    #sanity_check(env)

    #pbc=18 means that pos_bin_size = 0.1 and vbc=14 means vel_bin_size = 0.01. So i use em as a basis
    position_bins_count = 18*4
    velocity_bins_count = 14*4

    agent = MountainCarMonteCarlo(train_env, test_env, position_bins_count=18, velocity_bins_count=14, gamma=0.99, epsilon=0.1)

    start = time.time()
    rewards = agent.train_mc(episodes=50000)
    optimal_policy = agent.derive_policy()

    stop = time.time()
    print(f"Training finished. Time spent: {stop - start} seconds.")
    torch.save({"policy": optimal_policy}, "mc_optimal_policy.pth")

    #policy_obj = torch.load("mc_optimal_policy.pth", weights_only=False)
    #optimal_policy = policy_obj["policy"]

    # Test visually
    test_reward = agent.test_policy(optimal_policy, max_steps=2000, render=True)
    print("Actual Environment test reward:", test_reward)

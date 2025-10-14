import gymnasium as gym
import numpy as np

def sanity_check(env, num_steps=100, render=True):
    """Simple Gymnasium environment sanity check."""
    try:
        obs, info = env.reset(seed=42)

        print(f"Environment loaded successfully.")
        print(f"Initial observation: {obs}")
        print(f"Observation space: {env.unwrapped.observation_space}")
        print(f"Action space: {env.unwrapped.action_space}\n")

        for step in range(num_steps):
            if render:
                env.render()

            # Sample a random action
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            print(f"Step {step + 1}:")
            print(f"  Action: {action}")
            print(f"  Obs: {obs}")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}\n")

            if done:
                print("Episode finished early. Resetting environment.\n")
                obs, info = env.reset()

        env.close()
        print("Sanity check completed successfully.")

    except Exception as e:
        print(f"Error during sanity check: {e}")


class MountainCarIteration:
    def __init__(self, gym_env, position_bins_count, velocity_bins_count):
        self.pbc = position_bins_count
        self.vbc = velocity_bins_count
        self.env = gym_env

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

        bin_step = (high-low) / (bins_count-1)
        bin_idx = int(round((value-low) / bin_step))

        return int(np.clip(bin_idx, 0, bins_count-1))

    def convert_discrete_continuous(self, bin_idx, low, high, bins_count):
        if bins_count == 1:
            return (low+high) / 2

        return low + bin_idx * (high-low) / (bins_count-1)



    def predict_next_state(self, curr_state, action):
        g = self.env.unwrapped.gravity
        force = self.env.unwrapped.force

        pos_low, pos_high = self.env.observation_space.low[0], self.env.observation_space.high[0]
        vel_low, vel_high = self.env.observation_space.low[1], self.env.observation_space.high[1]

        pos, vel = curr_state

        vel_next = vel + (action-1)*force - g*np.cos(3*pos)
        vel_next = np.clip(vel_next, vel_low, vel_high)

        pos_next = pos + vel_next
        pos_next = np.clip(pos_next, pos_low, pos_high)

        return pos_next, vel_next


    def test_policy(self, policy, max_steps=1000, render=True):
        state, _ = self.env.reset()

        if render:
            self.env.render()

        total_reward = 0
        for _ in range(max_steps):
            pos_bin, vel_bin = self.discretize_state(state=state)

            action = int(policy[pos_bin, vel_bin])
            state, reward, terminated, truncated, info = self.env.step(action)

            total_reward += reward
            if render:
                self.env.render()
            if terminated:
                break

        return total_reward



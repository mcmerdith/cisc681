from argparse import ArgumentParser

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def action_to_text(action):
    """
    Convert action to text for FrozenLake environment.
    """
    if action == 0:
        return "LEFT"
    elif action == 1:
        return "DOWN"
    elif action == 2:
        return "RIGHT"
    elif action == 3:
        return "UP"
    else:
        raise ValueError("Invalid action")


def main(is_slippery: bool = True, success_rate: float = 0.8):
    # Initialise the environment
    env = gym.make(
        "FrozenLake-v1",
        render_mode="human",
        map_name="4x4",
        # desc=generate_random_map(size=4),
        is_slippery=is_slippery,
        success_rate=success_rate,
        reward_schedule=(10, -10, -0.001),
    )
    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    print("start state is ", observation)
    for _ in range(100):
        # this is where you would insert your policy
        action = env.action_space.sample()
        previous_state = observation
        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            "Transition (s,a,r,s'): ",
            previous_state,
            action_to_text(action),
            reward,
            observation,
        )
        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--is-slippery", action="store_true", dest="is_slippery")
    parser.add_argument("--success-rate", type=float, dest="success_rate", default=None)
    args = parser.parse_args()
    main(**{k: v for k, v in vars(args).items() if v is not None})

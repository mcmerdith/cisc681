import os
import pickle
from argparse import ArgumentParser
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium.core import Env
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def agent_file_path(name: str):
    base = os.path.join(os.getcwd(), "agents")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name + ".npy")


@dataclass
class Agent:
    world: Env
    world_size: tuple[int, int]
    position: int
    learning_rate: float = field(kw_only=True, default=0.8)
    discount_factor: float = field(kw_only=True, default=0.95)
    exploration_probability: float = field(kw_only=True, default=0.2)
    exploration_optimism: float = field(kw_only=True, default=1)
    """Higher values result in more exploration"""
    _q: np.ndarray = field(init=False)
    _n: np.ndarray = field(init=False)
    _random: np.random.Generator = field(init=False)

    def __post_init__(self):
        self._q = np.zeros(shape=((self.world_size[0] * self.world_size[1], 4)))
        self._n = np.ones(shape=((self.world_size[0] * self.world_size[1], 4)))
        self._random = self.world.np_random

    def learn(self, iterations: int):
        for _ in range(iterations):
            self.make_decision()

    def execute(self):
        while True:
            print(self._q[self.position])
            action = np.argmax(self._q[self.position])
            print(
                f"Moving {action_to_text(action)} from {self.position % self.world_size[0] + int(self.position / self.world_size[0])}",
            )
            observation, reward, terminated, truncated, info = self.world.step(action)
            self.position = observation
            if terminated or truncated:
                break

    def make_decision(self):
        if self._random.random() < self.exploration_probability:
            # random action
            action = self._random.integers(0, 3)
        else:
            # learned action
            action = np.argmax(self._q[self.position])

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = self.world.step(action)

        # max_a'[Q(s', a') + optimism / N(s', a')]
        best_next_action = np.argmax(
            self._q[observation] + self.exploration_optimism / self._n[observation]
        )
        self._n[self.position, action] += 1
        new_q = reward + self.discount_factor * self._q[observation, best_next_action]
        self._q[self.position, action] = (
            (1 - self.learning_rate) * self._q[self.position, action]
        ) + (self.learning_rate * new_q)
        self.position = observation
        print(
            "Transition (s,a,r,s'): ",
            self.position,
            action_to_text(action),
            reward,
            observation,
        )
        print(
            "Learned that (",
            self.position,
            ",",
            action,
            ") =",
            self._q[self.position, action],
        )

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = self.world.reset()
            self.position = observation

        if self.learning_rate >= 0.01:
            self.learning_rate -= 0.01
        if self.exploration_probability >= 0.005:
            self.exploration_probability -= 0.005

    def save_to_file(self, name: str) -> None:
        np.save(agent_file_path(name), self._q)

    def load_from_file(self, name: str) -> None:
        self._q = np.load(agent_file_path(name))


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


def main(
    is_slippery: bool = True,
    success_rate: float = 0.8,
    iterations: int = 1000,
    filename: str = "agent",
):
    filename = filename + "_" + str(iterations)
    load = os.path.exists(agent_file_path(filename))

    # Initialise the environment
    env = gym.make(
        "FrozenLake-v1",
        render_mode="human" if load else None,
        map_name="4x4",
        # desc=generate_random_map(size=4),
        is_slippery=is_slippery,
        success_rate=success_rate,
        reward_schedule=(10, -10, -0.001),
    )
    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    print("start state is", observation)

    agent = Agent(env, (4, 4), observation)
    if load:
        print("Loading saved agent:", filename)
        agent.load_from_file(filename)
        agent.exploration_probability = 0

    if load:
        agent.execute()
    else:
        agent.learn(iterations)
        print(agent._q)
        agent.save_to_file(filename)
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    # program arguments
    parser.add_argument("--is-slippery", action="store_true", dest="is_slippery")
    parser.add_argument("--success-rate", type=float, dest="success_rate", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()
    # pass through only the arguments which have values
    main(**{k: v for k, v in vars(args).items() if v is not None})

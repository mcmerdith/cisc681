import os
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
    world_size: tuple[int, int]

    # learning parameters
    learning_rate: float = field(kw_only=True, default=0.8)
    discount_factor: float = field(kw_only=True, default=0.95)
    exploration_probability: float = field(kw_only=True, default=0.2)
    exploration_optimism: float = field(kw_only=True, default=1)
    """Higher values result in more exploration"""

    _world: Env = field(init=False)
    _q: np.ndarray = field(init=False)
    _n: np.ndarray = field(init=False)
    _random: np.random.Generator = field(init=False)

    def __post_init__(self):
        self._q = np.zeros(shape=((self.world_size[0] * self.world_size[1], 4)))
        self._n = np.ones(shape=((self.world_size[0] * self.world_size[1], 4)))

    def learn(self, max_iterations: int):
        # reduce the learning parameters towards 0 at the end of training
        learning_anneal = self.learning_rate / max_iterations
        probability_anneal = self.exploration_probability / max_iterations
        for i in range(max_iterations):
            self.make_decision()
            if self.solved():
                print(f"solved in {i} iterations")
                break

            # reduce learning parameters on each iteration
            if self.learning_rate >= learning_anneal:
                self.learning_rate -= learning_anneal
            if self.exploration_probability >= probability_anneal:
                self.exploration_probability -= probability_anneal

    def execute(self):
        while True:
            action = np.argmax(self._q[self.position])
            observation, _, terminated, truncated, _ = self._world.step(action)
            self.position = observation
            if terminated or truncated:
                break

    def make_decision(self):
        if self._random.random() < self.exploration_probability:
            # random action
            action = self._world.action_space.sample()
        else:
            # learned action
            action = np.argmax(self._q[self.position])

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, _ = self._world.step(action)

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

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = self._world.reset()
            self.position = observation

    def solved(self) -> bool:
        return bool(np.any(self._q[0] > 0))

    def reset(self, world: Env, seed: int | None = None):
        self._world = world
        self._random = self._world.np_random
        self.position, _ = world.reset(seed=seed)

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
    world_size: int,
    random_world: bool = False,
    is_slippery: bool = True,
    success_rate: float = 0.8,
    max_iterations: int = 10000,
):
    if random_world:
        spec = {"desc": generate_random_map(size=world_size)}
    else:
        spec = {"map_name": f"{world_size}x{world_size}"}

    def create_world(**opts):
        return gym.make(
            "FrozenLake-v1",
            **spec,
            is_slippery=is_slippery,
            success_rate=success_rate,
            reward_schedule=(10, -10, -0.001),
            **opts,
        )

    agent = Agent((world_size, world_size))

    # Train
    with create_world(render_mode=None) as env:
        agent.reset(env, 42)
        agent.learn(max_iterations)

    # Test
    with create_world(render_mode="human") as env:
        agent.reset(env, 42)
        agent.execute()


if __name__ == "__main__":
    parser = ArgumentParser()
    # program arguments
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--random-world", action="store_true")
    parser.add_argument("--is-slippery", action="store_true")
    parser.add_argument("--success-rate", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()
    # pass through only the arguments which have values
    main(**{k.replace("-", "_"): v for k, v in vars(args).items() if v is not None})

from argparse import ArgumentParser
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from gymnasium.core import Env
from tqdm import tqdm

from utils import (
    CONVERGENCE_METHOD_NAMES,
    AgentBatchPerformance,
    AgentPerformance,
    ConvergenceMethod,
    ConvergenceMethodName,
    agent_file_path,
    create_spec,
    create_world,
    e_decay,
)


@dataclass
class Agent:
    world: Env
    convergence_method: ConvergenceMethod = field(
        default_factory=lambda: ConvergenceMethod("policy_convergence")
    )

    # average-case default learning parameters
    learning_rate: float = field(kw_only=True, default=0.3)
    """Higher values result in faster learning"""
    exploration_prob: float = field(kw_only=True, default=0.3)
    """Higher values result in more exploration"""
    optimism: float = field(kw_only=True, default=1)
    """Higher values result in more exploration"""
    discount_factor: float = field(kw_only=True, default=0.95)
    """Higher values prioritize long-term rewards"""

    # internal agent state
    _world_size: int = field(init=False)
    _position: int = field(init=False)

    _q: np.ndarray = field(init=False)
    _n: np.ndarray = field(init=False)

    _iteration: int = field(init=False, default=0)

    # convergence params
    _q_deltas: deque = field(init=False)
    _stable_policy_iterations: int = field(init=False, default=0)

    def __post_init__(self):
        self.reset_world()
        self.reset_internals()

    def reset_world(self, world: Env | None = None, seed: int | None = None):
        if world is not None:
            self.world = world
        self._world_size = int(self.world.observation_space.n)  # pyright: ignore[reportAttributeAccessIssue]
        self._position, _ = self.world.reset(seed=seed)

    def reset_internals(self):
        self._q = np.zeros(shape=((self._world_size, 4)))
        self._n = np.ones(shape=((self._world_size, 4)))
        self._iteration = 0
        self._q_deltas = deque(maxlen=self.convergence_method.stable_window_size)
        self._stable_policy_iterations = 0

    def get_policy(self) -> np.ndarray:
        return np.argmax(self._q, axis=1)

    def execute(self) -> "AgentPerformance":
        """
        Execute a greedy solution (no exploration or learning)

        Returns AgentPerformance
        """
        performance = AgentPerformance(self.world)
        while True:
            action = np.argmax(self._q[self._position])
            observation, _, terminated, truncated, info = self.world.step(action)
            performance.step(observation, terminated, truncated, info)
            if self._iteration % 1000 == 0:
                performance.snapshot(self._q_deltas)
            self._position = observation
            if terminated or truncated:
                return performance

    def learn(self, max_iterations: int) -> Literal[False] | int:
        """
        Learn Q-values for the current world

        Returns the iteration the agent converged on, or False otherwise
        """
        learning_rate = self.learning_rate
        exploration_prob = self.exploration_prob
        optimism = self.optimism

        policy = self.get_policy()

        for self._iteration in range(max_iterations):
            if np.random.random() < exploration_prob:
                # random action
                action = self.world.action_space.sample()
            else:
                # learned action
                action = np.argmax(self._q[self._position])

            # step (transition) through the environment with the action
            # receiving the next observation, reward and if the episode has terminated or truncated
            observation, reward, terminated, truncated, _ = self.world.step(action)
            reward = float(reward)  # shutup type checker

            if self.optimism == 0:
                # take the logical option
                new_q = reward + self.discount_factor * np.max(self._q[observation])
            else:
                # encourage optimism
                # max_a'[Q(s', a') + optimism / N(s', a')]
                optimistic_next_action = np.argmax(
                    self._q[observation] + optimism / self._n[observation]
                )
                new_q = (
                    reward
                    + self.discount_factor
                    * self._q[observation, optimistic_next_action]
                )

            prev_q = self._q[self._position, action]
            next_q = ((1 - learning_rate) * prev_q) + (learning_rate * new_q)

            # save the delta to the window
            self._q_deltas.append(np.abs(next_q - prev_q))
            # update the Q and N tables
            self._q[self._position, action] = next_q
            self._n[self._position, action] += 1

            # update agent position
            self._position = observation

            # check for a stable policy
            new_policy = self.get_policy()
            if all(policy == new_policy):
                self._stable_policy_iterations += 1
            else:
                policy = new_policy
                self._stable_policy_iterations = 0

            # check if the agent has learned
            if self.converged():
                return self._iteration

            # If the episode has ended then we can reset to start a new episode
            if terminated or truncated:
                self.reset_world()

            # slowly reduce learning rate on each iteration
            learning_rate = e_decay(self.learning_rate, 0.1, 0.001, self._iteration)
            # quickly reduce our exploration rate
            exploration_prob = e_decay(
                self.exploration_prob, 0.0, 0.005, self._iteration
            )

        return False

    def converged(self):
        if (
            self.convergence_method.name is None
            or self._iteration < self.convergence_method.stable_window_size
        ):
            return False

        # if we found a path to the goal, there will be a reward at the start
        if np.max(self._q[0]) <= 0:
            return False

        if self.convergence_method.name == "q_convergence":
            # check if the q values have stabilized
            if np.average(self._q_deltas) < self.convergence_method.q_delta_threshold:
                return True
        elif self.convergence_method.name == "policy_convergence":
            # check if the policy is stabilized
            if (
                self._stable_policy_iterations
                > self.convergence_method.stable_window_size
            ):
                return True
        return False

    def save_to_file(self, name: str) -> None:
        np.save(agent_file_path(name), self._q)

    def load_from_file(self, name: str) -> None:
        self._q = np.load(agent_file_path(name))


if __name__ == "__main__":
    parser = ArgumentParser()
    # world params arguments
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--random-world", action="store_true")
    parser.add_argument("--success-rate", type=float, default=1)
    # agent parameters and hyperparameters
    parser.add_argument(
        "--convergence-method",
        choices=CONVERGENCE_METHOD_NAMES,
        default=None,
    )
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--exploration-prob", type=float, default=None)
    parser.add_argument("--optimism", type=int, default=None)
    parser.add_argument("--discount-factor", type=float, default=None)
    # utils
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--test-agent", action="store_true")

    args = parser.parse_args()

    # extract args
    world_size: int = args.world_size
    random_world: bool = args.random_world
    success_rate: float = args.success_rate
    slippery: bool = success_rate < 1

    convergence_method: ConvergenceMethodName | None = args.convergence_method
    max_iterations: int = args.max_iterations
    learning_rate: float | None = args.learning_rate
    exploration_prob: float | None = args.exploration_prob
    optimism: int | None = args.optimism
    discount_factor: float | None = args.discount_factor

    print(
        f"Agent will operate in a {world_size}x{world_size}",
        "stochastic" if slippery else "deterministic",
        f"world with a success rate of {success_rate}",
    )
    if convergence_method is not None:
        print(f"Agent convergence will be evaluated with {convergence_method}")

    spec = create_spec(world_size, random_world)
    world = create_world(spec, slippery, success_rate)

    agent = Agent(
        world,
        ConvergenceMethod(convergence_method),
        **{
            k: v
            for k, v in [
                ("learning_rate", learning_rate),
                ("exploration_prob", exploration_prob),
                ("optimism", optimism),
                ("discount_factor", discount_factor),
            ]
            if v is not None
        },
    )

    # Train
    with world:
        agent.reset_world(world, 42)
        print("Converged?", agent.learn(max_iterations))

    if args.test_agent:
        performance = AgentBatchPerformance()
        for i in tqdm(range(2000), "Testing agent performance"):
            world = create_world(spec, slippery, success_rate)
            with world:
                agent.reset_world(world)
                performance.accumulate(agent.execute())

        print(performance)

    if args.visualize:
        while True:
            world = create_world(spec, slippery, success_rate, render_mode="human")
            with world:
                agent.reset_world(world)
                if agent.execute().success:
                    break

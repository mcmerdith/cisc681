from argparse import ArgumentParser
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from gymnasium.core import Env
from tqdm import tqdm

from utils import (
    CONVERGENCE_METHODS,
    AgentPerformance,
    ConvergenceMethod,
    agent_file_path,
    create_spec,
    create_world,
    e_decay,
    p_decay,
)


@dataclass
class ConvergenceParams:
    q_window_size: int = 500
    q_delta_threshold: float = 0.01

    stable_policy_iterations: int = 500

    test_sample_size: int = 1000
    test_success_threshold: float = 0.85


@dataclass
class Agent:
    world: Env
    convergence_methods: list[ConvergenceMethod] | None
    convergence_params: ConvergenceParams = field(default_factory=ConvergenceParams)

    # learning parameters
    learning_rate: float = field(kw_only=True, default=0.8)
    exploration_prob: float = field(kw_only=True, default=0.2)
    """Higher values result in more exploration"""
    optimism: float = field(kw_only=True, default=1)
    """Higher values result in more exploration"""
    discount_factor: float = field(kw_only=True, default=0.95)
    """Higher values prioritize long-term rewards"""

    # internal agent state
    _world_size: int = field(init=False)
    _position: int = field(init=False)

    _q: np.ndarray = field(init=False)
    _q_deltas: deque = field(init=False)

    _n: np.ndarray = field(init=False)

    _iteration: int = field(init=False, default=0)
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
        self._q_deltas = deque(maxlen=self.convergence_params.q_window_size)
        self._n = np.ones(shape=((self._world_size, 4)))
        self._iteration = 0
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

    def learn(self, max_iterations: int):
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
                return True

            # If the episode has ended then we can reset to start a new episode
            if terminated or truncated:
                self.reset_world()

            # slowly reduce learning rate on each iteration
            learning_rate = p_decay(
                self.learning_rate, 0.001, self._iteration, max_iterations
            )
            # quickly reduce our exploration rate
            exploration_prob = e_decay(
                self.exploration_prob, 0.0, 0.995, self._iteration
            )

        return False

    def converged(self):
        if self.convergence_methods is None or len(self.convergence_methods) == 0:
            return False
        # if we found a path to the goal, there will be a reward at the start
        if np.max(self._q[0]) <= 0:
            return False
        learned = True
        for method in self.convergence_methods:
            if method == "q_convergence":
                # check if the q values have stabilized
                if self._iteration < self.convergence_params.q_window_size:
                    learned = False
                if (
                    np.average(self._q_deltas)
                    > self.convergence_params.q_delta_threshold
                ):
                    learned = False
            elif method == "policy_convergence":
                # check if the policy is stabilized
                if self._iteration < self.convergence_params.stable_policy_iterations:
                    learned = False
                if (
                    self._stable_policy_iterations
                    < self.convergence_params.stable_policy_iterations
                ):
                    learned = False
            elif method == "test_success":
                # check if the policy results in a success
                success = 0
                for _ in range(self.convergence_params.test_sample_size):
                    self.reset_world()
                    if self.execute().success:
                        success += 1
                self.reset_world()
                if (
                    success / self.convergence_params.test_sample_size
                    < self.convergence_params.test_success_threshold
                ):
                    learned = False
        return learned

    def save_to_file(self, name: str) -> None:
        np.save(agent_file_path(name), self._q)

    def load_from_file(self, name: str) -> None:
        self._q = np.load(agent_file_path(name))


def train_agent(
    world: Env,
    convergence_methods: list[ConvergenceMethod] | None,
    max_iterations: int,
):
    agent = Agent(
        world,
        convergence_methods=convergence_methods,
    )

    # Train
    with world:
        agent.reset_world(world, 42)
        agent.learn(max_iterations)

    return agent


if __name__ == "__main__":
    parser = ArgumentParser()
    # program arguments
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--random-world", action="store_true")
    parser.add_argument("--slippery", action="store_true")
    parser.add_argument("--success-rate", type=float, default=0.75)
    parser.add_argument(
        "--convergence-methods", choices=CONVERGENCE_METHODS, nargs="*", default=None
    )
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--test-agent", action="store_true")

    args = parser.parse_args()

    world_size: int = args.world_size
    random_world: bool = args.random_world
    slippery: bool = args.slippery
    success_rate: float = args.success_rate if slippery else 1
    convergence_methods: list[ConvergenceMethod] | None = args.convergence_methods
    max_iterations: int = args.max_iterations

    print(
        f"Agent will operate in a {world_size}x{world_size}",
        "stochastic" if slippery else "deterministic",
        f"world with a success rate of {success_rate}",
    )
    if convergence_methods is not None and len(convergence_methods) > 0:
        print(
            f"Agent convergence will be evaluated with {' + '.join(convergence_methods)}"
        )

    spec = create_spec(world_size, random_world)
    world = create_world(spec, slippery, success_rate)

    agent = train_agent(world, convergence_methods, max_iterations)

    if args.test_agent:
        success = 0
        bad_luck = 0
        out_of_moves = 0
        for i in tqdm(range(2000), "Testing agent performance"):
            world = create_world(spec, slippery, success_rate)
            with world:
                agent.reset_world(world)
                result = agent.execute()
                if result.success:
                    success += 1
                else:
                    if result.unlucky:
                        bad_luck += 1
                    elif result.out_of_moves:
                        out_of_moves += 1

        print(f"Agent reached the goal: {success / 20}%")
        print(f"Agent died of bad luck: {bad_luck / 20}%")
        print(f"Agent ran out of moves: {out_of_moves / 20}%")

    if args.visualize:
        while True:
            world = create_world(spec, slippery, success_rate, render_mode="human")
            with world:
                agent.reset_world(world)
                if agent.execute().success:
                    break

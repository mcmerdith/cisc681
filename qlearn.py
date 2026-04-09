import os
from argparse import ArgumentParser
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium.core import Env
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from tqdm import tqdm


def agent_file_path(name: str):
    base = os.path.join(os.getcwd(), "agents")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name + ".npy")


EVALUATION_METHODS = ["q_convergence", "policy_convergence", "test_success"]
EvaluationMethod = (
    Literal["q_convergence"] | Literal["policy_convergence"] | Literal["test_success"]
)


@dataclass
class EvaluationParams:
    q_window_size: int = 500
    q_delta_threshold: float = 0.01

    stable_policy_iterations: int = 500

    test_sample_size: int = 1000
    test_success_threshold: float = 0.85


@dataclass
class AgentPerformance:
    agent: "Agent"
    success: bool = False
    out_of_moves: bool = False
    unlucky: bool = False
    total_moves: int = 0
    failed_moves: int = 0
    q_delta_snapshots: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.world = agent.world

    def step(self, observation: int, terminated: bool, truncated: bool, info: dict):
        self.total_moves += 1

        if terminated:
            self.success = observation == agent.world_size - 1
            self.out_of_moves = truncated

        if (
            self.world.spec is not None
            and info["prob"] != self.world.spec.kwargs["success_rate"]
        ):
            self.failed_moves += 1
            if terminated:
                self.unlucky = True

        if self.agent._iteration % 1000 == 0:
            # take a snapshot of the parameters
            self.q_delta_snapshots.append(np.average(self.agent._q_deltas))
            pass


@dataclass
class Agent:
    world: Env
    world_size: int = field(init=False)
    evaluation_methods: list[EvaluationMethod] | None
    evaluation_params: EvaluationParams = field(default_factory=EvaluationParams)
    seed: int | None = None

    # learning parameters
    learning_rate: float = field(kw_only=True, default=0.8)
    exploration_prob: float = field(kw_only=True, default=0.2)
    """Higher values result in more exploration"""
    optimism: float = field(kw_only=True, default=1)
    """Higher values result in more exploration"""
    discount_factor: float = field(kw_only=True, default=0.95)
    """Higher values prioritize long-term rewards"""

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
        if seed is not None:
            self.seed = seed
        if self.world.spec is None:
            raise RuntimeError()
        self.world_size = int(self.world.spec.kwargs["size"]) ** 2
        print(self.world_size)
        self.position, _ = self.world.reset(seed=self.seed)

    def reset_internals(self):
        self._q = np.zeros(shape=((self.world_size, 4)))
        self._q_deltas = deque(maxlen=self.evaluation_params.q_window_size)
        self._n = np.ones(shape=((self.world_size, 4)))
        self._iteration = 0
        self._stable_policy_iterations = 0

    def get_policy(self) -> np.ndarray:
        return np.argmax(self._q, axis=1)

    def execute(self, print_info: bool = False) -> AgentPerformance:
        """
        Execute a greedy solution (no exploration or learning)

        Returns
            [0]: True if the goal was reached
            [1]: True if the agent failed it's last move
            [2]: True if the agent ran out of moves
        """
        performance = AgentPerformance(self)
        while True:
            action = np.argmax(self._q[self.position])
            observation, _, terminated, truncated, info = self.world.step(action)
            performance.step(observation, terminated, truncated, info)
            self.position = observation
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
                action = np.argmax(self._q[self.position])

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
                explore_f = self._q[observation] + optimism / self._n[observation]
                explore_next_action = np.argmax(explore_f)
                new_q = (
                    reward
                    + self.discount_factor * self._q[observation, explore_next_action]
                )

            prev_q = self._q[self.position, action]
            next_q = ((1 - learning_rate) * prev_q) + (learning_rate * new_q)

            # save the delta to the window
            self._q_deltas.append(np.abs(next_q - prev_q))
            # update the Q and N tables
            self._q[self.position, action] = next_q
            self._n[self.position, action] += 1

            # update agent position
            self.position = observation

            # check for a stable policy
            new_policy = self.get_policy()
            if all(policy == new_policy):
                self._stable_policy_iterations += 1
            else:
                policy = new_policy
                self._stable_policy_iterations = 0

            # check if the agent has learned
            if self.has_learned():
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

    def has_learned(self):
        if self.evaluation_methods is None or len(self.evaluation_methods) == 0:
            return False
        # there needs to be a reward at the start location
        if np.max(self._q[0]) <= 0:
            return False
        learned = True
        for method in self.evaluation_methods:
            if method == "q_convergence":
                # check if the q values have stabilized
                if self._iteration < self.evaluation_params.q_window_size:
                    learned = False
                if (
                    np.average(self._q_deltas)
                    > self.evaluation_params.q_delta_threshold
                ):
                    learned = False
            elif method == "policy_convergence":
                # check if the policy is stabilized
                if self._iteration < self.evaluation_params.stable_policy_iterations:
                    learned = False
                if (
                    self._stable_policy_iterations
                    < self.evaluation_params.stable_policy_iterations
                ):
                    learned = False
            elif method == "test_success":
                # check if the policy results in a success
                success = 0
                for _ in range(self.evaluation_params.test_sample_size):
                    self.reset_world()
                    if self.execute().success:
                        success += 1
                self.reset_world()
                if (
                    success / self.evaluation_params.test_sample_size
                    < self.evaluation_params.test_success_threshold
                ):
                    learned = False
        return learned

    def save_to_file(self, name: str) -> None:
        np.save(agent_file_path(name), self._q)

    def load_from_file(self, name: str) -> None:
        self._q = np.load(agent_file_path(name))


def l_decay(initial: float, min: float, iteration: int, max_iteration: int):
    return max(min, (initial - min) * (iteration / max_iteration))


def p_decay(initial: float, min: float, iteration: int, max_iteration: int):
    return min + (initial - min) * (1 - (iteration / max_iteration) ** 3)


def e_decay(initial: float, min: float, rate: float, iteration: int):
    return min + (initial - min) * np.exp(-rate * iteration)


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


def create_spec(world_size: int, random_world: bool):
    if random_world:
        return {"size": world_size, "desc": generate_random_map(size=world_size)}
    else:
        return {"size": world_size, "map_name": f"{world_size}x{world_size}"}


def create_world(
    world_spec: dict,
    is_slippery: bool,
    success_rate: float,
    reward_schedule: tuple[float, float, float] = (10, -10, -0.001),
    render_mode: None | Literal["human"] = None,
    **opts,
):
    return gym.make(
        "FrozenLake-v1",
        **world_spec,
        is_slippery=is_slippery,
        success_rate=success_rate,
        reward_schedule=(10, -10, -0.001),
        # reward_schedule=(1, -1, 0),
        render_mode=render_mode,
        **opts,
    )


def train_agent(
    world: Env,
    evaluation_methods: list[EvaluationMethod] | None,
    max_iterations: int,
):
    agent = Agent(
        world,
        evaluation_methods=evaluation_methods,
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
        "--evaluation-methods", choices=EVALUATION_METHODS, nargs="*", default=None
    )
    parser.add_argument("--max-iterations", type=int, default=10000)

    args = parser.parse_args()

    world_size: int = args.world_size
    random_world: bool = args.random_world
    slippery: bool = args.slippery
    success_rate: float = args.success_rate if slippery else 1
    evaluation_methods: list[EvaluationMethod] | None = args.evaluation_methods
    max_iterations: int = args.max_iterations

    print(
        f"Agent will operate in a {world_size}x{world_size}",
        "stochastic" if slippery else "deterministic",
        f"world with a success rate of {success_rate}",
    )
    if args.evaluation_methods is not None and len(args.evaluation_methods) > 0:
        print(f"Agent will be evaluated with {' + '.join(args.evaluation_methods)}")

    spec = create_spec(world_size, random_world)
    world = create_world(spec, slippery, success_rate)

    agent = train_agent(world, evaluation_methods, max_iterations)

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

    while True:
        world = create_world(spec, slippery, success_rate, render_mode="human")
        with world:
            agent.reset_world(world)
            if agent.execute().success:
                break

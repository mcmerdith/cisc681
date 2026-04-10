import os
from dataclasses import dataclass, field
from typing import Literal, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import Env
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def agent_file_path(name: str):
    base = os.path.join(os.getcwd(), "agents")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name + ".npy")


ConvergenceMethodName = Literal["q_convergence"] | Literal["policy_convergence"]
CONVERGENCE_METHOD_NAMES: list[ConvergenceMethodName] = [
    "q_convergence",
    "policy_convergence",
]


def create_spec(world_size: int, random_world: bool):
    if random_world:
        return {"desc": generate_random_map(size=world_size)}
    else:
        return {"map_name": f"{world_size}x{world_size}"}


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
        reward_schedule=reward_schedule,
        render_mode=render_mode,
        **opts,
    )


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


def l_decay(initial: float, min: float, iteration: int, max_iteration: int):
    return max(min, (initial - min) * (iteration / max_iteration))


def p_decay(
    initial: float, min: float, iteration: int, max_iteration: int, power: int = 2
):
    return max(min, initial * (1 - (iteration / max_iteration) ** power))


def e_decay(initial: float, min: float, rate: float, iteration: int):
    return max(min, initial * np.exp(-rate * iteration))


def padded_str(o, length: int = 5):
    val = str(o)
    if len(val) < length:
        val = " " * (length - len(val)) + val
    return val


def padded_number(value: float | int, precision: int = 2, length: int = 5):
    return padded_str(round(value, precision))


def percent(value: float):
    return padded_number(value * 100) + "%"


@dataclass
class AgentPerformance:
    world: Env

    success: bool = False
    out_of_moves: bool = False
    unlucky: bool = False
    total_moves: int = 0
    failed_moves: int = 0
    q_delta_snapshots: list[float] = field(default_factory=list)

    _success_rate: float | None = field(init=False, default=None)
    _world_size: int = field(init=False)

    def __post_init__(self):
        self._world_size = int(self.world.observation_space.n)  # pyright: ignore[reportAttributeAccessIssue]
        if self.world.spec is not None:
            self._success_rate = self.world.spec.kwargs["success_rate"]

    def step(self, observation: int, terminated: bool, truncated: bool, info: dict):
        self.total_moves += 1

        if terminated:
            self.success = observation == self._world_size - 1
        if truncated:
            self.out_of_moves = truncated

        if info["prob"] != self._success_rate:
            self.failed_moves += 1
            if terminated and not self.success:
                self.unlucky = True

    def snapshot(self, q_deltas: Sequence):
        self.q_delta_snapshots.append(np.average(q_deltas))


@dataclass
class AgentBatchPerformance:
    samples: int = 0
    success_count: float = 0
    out_of_moves_count: float = 0
    unlucky_count: float = 0
    total_moves: float = 0
    failed_moves: float = 0

    def average(self, value: float):
        if self.samples == 0:
            return 0
        else:
            return value / self.samples

    def accumulate(self, performance: AgentPerformance):
        self.samples += 1
        if performance.success:
            self.success_count += 1
        if performance.out_of_moves:
            self.out_of_moves_count += 1
        if performance.unlucky:
            self.unlucky_count += 1
        self.total_moves += performance.total_moves
        self.failed_moves += performance.failed_moves

    def __str__(self):
        return (
            f"Success: {percent(self.average(self.success_count))}"
            + f"  Out of Moves: {percent(self.average(self.out_of_moves_count))}"
            + f"  Unlucky: {percent(self.average(self.unlucky_count))}"
            + f"  Average Total Moves: {padded_number(self.average(self.total_moves))}"
            + f"  Average Failed Moves: {padded_number(self.average(self.failed_moves))}"
        )

    def __lt__(self, other: "AgentBatchPerformance"):
        return (self.average(self.success_count)) < (other.average(other.success_count))

    def __gt__(self, other: "AgentBatchPerformance"):
        return (self.average(self.success_count)) > (other.average(other.success_count))


@dataclass
class WorldParameters:
    world_size: int
    random_world: bool
    slippery: bool
    success_rate: float
    reward_schedule: tuple[float, float, float]

    @staticmethod
    def combinations(
        world_sizes: list[int] | None = None,
        random_worlds: list[bool] | None = None,
        success_rates: list[float] | None = None,
        reward_schedules: list[tuple[float, float, float]] | None = None,
    ):
        if world_sizes is None:
            world_sizes = [4, 8]
        if random_worlds is None:
            random_worlds = [False, True]
        if success_rates is None:
            success_rates = [0.75, 1]
        if reward_schedules is None:
            reward_schedules = [(10, -10, -0.001)]
        combinations: list[WorldParameters] = []
        for world_size in world_sizes:
            for random_world in random_worlds:
                for success_rate in success_rates:
                    for reward_schedule in reward_schedules:
                        combinations.append(
                            WorldParameters(
                                world_size,
                                random_world,
                                success_rate < 1,
                                success_rate,
                                reward_schedule,
                            )
                        )
        return combinations

    def __str__(self):
        return (
            ("Random" if self.random_world else "Static")
            + f" {self.world_size}x{self.world_size} "
            + ("stochastic world   " if self.slippery else "deterministic world")
            + f"  (success rate {percent(self.success_rate)})"
            + f"  Rewards: {padded_str(self.reward_schedule, 17)}"
        )

    def __lt__(self, other: "WorldParameters"):
        return (
            self.world_size,
            self.random_world,
            self.slippery,
            self.success_rate,
            self.reward_schedule,
        ) < (
            other.world_size,
            other.random_world,
            other.slippery,
            other.success_rate,
            other.reward_schedule,
        )


@dataclass
class ConvergenceMethod:
    name: ConvergenceMethodName | None = None

    stable_window_size: int = field(kw_only=True, default=500)
    q_delta_threshold: float = field(kw_only=True, default=0.01)

    @staticmethod
    def combinations(
        convergence_methods: list[ConvergenceMethodName] | None = None,
        stable_window_sizes: list[int] | None = None,
        q_delta_thresholds: list[float] | None = None,
        always_include_none=False,
    ):
        if convergence_methods is None:
            convergence_methods = CONVERGENCE_METHOD_NAMES
        if stable_window_sizes is None:
            stable_window_sizes = [500, 1000]
        if q_delta_thresholds is None:
            q_delta_thresholds = [0.01, 0.02]
        combinations: list[ConvergenceMethod] = []
        for method in convergence_methods:
            for stable_window_size in stable_window_sizes:
                if method == "q_convergence":
                    for q_delta_threshold in q_delta_thresholds:
                        combinations.append(
                            ConvergenceMethod(
                                "q_convergence",
                                stable_window_size=stable_window_size,
                                q_delta_threshold=q_delta_threshold,
                            )
                        )
                elif method == "policy_convergence":
                    combinations.append(
                        ConvergenceMethod(
                            "policy_convergence", stable_window_size=stable_window_size
                        )
                    )
        if len(convergence_methods) == 0 or always_include_none:
            combinations.append(ConvergenceMethod(None))
        return combinations


@dataclass
class AgentParameters:
    max_iterations: int = 20000
    learning_rate: float = 0.3
    exploration_prob: float = 0.3
    optimism: float = 1
    discount_factor: float = 0.95
    convergence_method: ConvergenceMethod = field(
        default_factory=lambda: ConvergenceMethod("policy_convergence")
    )

    @staticmethod
    def combinations(
        max_iterations_l: list[int] | None = None,
        learning_rates: list[float] | None = None,
        exploration_probs: list[float] | None = None,
        optimisms: list[int] | None = None,
        discount_factors: list[float] | None = None,
        convergence_methods: list[ConvergenceMethod] | None = None,
    ):
        if max_iterations_l is None:
            max_iterations_l = [5000, 10000]
        if learning_rates is None:
            learning_rates = [0.2, 0.4, 0.6, 0.8]
        if exploration_probs is None:
            exploration_probs = [0.2, 0.4, 0.6, 0.8]
        if optimisms is None:
            optimisms = [1, 2]
        if discount_factors is None:
            discount_factors = [1, 0.995, 0.95, 0.85, 0.75]
        if convergence_methods is None:
            convergence_methods = [ConvergenceMethod(None)]
        combinations: list[AgentParameters] = []
        for max_iterations in max_iterations_l:
            for learning_rate in learning_rates:
                for exploration_prob in exploration_probs:
                    for optimism in optimisms:
                        for discount_factor in discount_factors:
                            for convergence_method in convergence_methods:
                                combinations.append(
                                    AgentParameters(
                                        max_iterations,
                                        learning_rate,
                                        exploration_prob,
                                        optimism,
                                        discount_factor,
                                        convergence_method,
                                    )
                                )
        return combinations

    def __str__(self):
        return (
            f"At most {padded_number(self.max_iterations)} iterations to "
            + padded_str(self.convergence_method.name or "completion")
            + f"  Learning Rate: {padded_number(self.learning_rate)}"
            + f"  Exploration: {percent(self.exploration_prob)}"
            + f"  Optimism: {padded_number(self.optimism)}"
            + f"  Discount Factor: {percent(self.discount_factor)}"
            + f"  Stable Iterations: {self.convergence_method.stable_window_size}"
            + f"  Q-delta Threshold: {self.convergence_method.q_delta_threshold}"
        )

    def __lt__(self, other: "AgentParameters"):
        return (
            self.learning_rate,
            self.max_iterations,
            self.exploration_prob,
            self.optimism,
            self.discount_factor,
            self.convergence_method.name,
        ) < (
            other.max_iterations,
            other.learning_rate,
            other.exploration_prob,
            other.optimism,
            other.discount_factor,
            other.convergence_method.name,
        )

import os
from collections import deque
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


CONVERGENCE_METHODS = ["q_convergence", "policy_convergence", "test_success"]
ConvergenceMethod = (
    Literal["q_convergence"] | Literal["policy_convergence"] | Literal["test_success"]
)


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
        reward_schedule=(10, -10, -0.001),
        # reward_schedule=(1, -1, 0),
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
    initial: float, min: float, iteration: int, max_iteration: int, power: int = 3
):
    return min + (initial - min) * (1 - (iteration / max_iteration) ** power)


def e_decay(initial: float, min: float, rate: float, iteration: int):
    return min + (initial - min) * np.exp(-rate * iteration)


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
        self._world_size = self.world.observation_space.n  # pyright: ignore[reportAttributeAccessIssue]
        if self.world.spec is not None:
            self._success_rate = self.world.spec.kwargs["success_rate"]

    def step(self, observation: int, terminated: bool, truncated: bool, info: dict):
        self.total_moves += 1

        if terminated:
            self.success = observation == self._world_size - 1
            self.out_of_moves = truncated

        if info["prob"] != self._success_rate:
            self.failed_moves += 1
            if terminated and not self.success:
                self.unlucky = True

    def snapshot(self, q_deltas: Sequence):
        self.q_delta_snapshots.append(np.average(q_deltas))


@dataclass
class AgentBatchPerformance:
    # performance
    samples: int = 0
    success_count: float = 0
    out_of_moves_count: float = 0
    unlucky_count: float = 0
    total_moves: float = 0
    failed_moves: float = 0

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
            f"Success: {percent(self.success_count / self.samples)}"
            + f"  Out of Moves: {percent(self.out_of_moves_count / self.samples)}"
            + f"  Unlucky: {percent(self.unlucky_count / self.samples)}"
            + f"  Average Total Moves: {padded_number(self.total_moves / self.samples)}"
            + f"  Average Failed Moves: {padded_number(self.failed_moves / self.samples)}"
            + f"  Average Failure Rate: {percent(self.failed_moves / self.total_moves)}"
        )

    def __lt__(self, other: "AgentBatchPerformance"):
        return (self.success_count / self.samples) < (
            other.success_count / other.samples
        )


@dataclass
class WorldParameters:
    world_size: int
    random_world: bool
    slippery: bool
    success_rate: float
    reward_schedule: tuple[float, float, float]

    @staticmethod
    def combinations():
        combinations = []
        for world_size in [4, 8]:
            for random_world in [True, False]:
                for success_rate in [0.75, 0.85, 0.95, 1]:
                    for reward_schedule in [(-10, 10, -0.001)]:
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
            self.slippery,
            self.success_rate,
            self.random_world,
            self.world_size,
            self.reward_schedule,
        ) < (
            other.slippery,
            other.success_rate,
            other.random_world,
            other.world_size,
            other.reward_schedule,
        )


@dataclass
class AgentParameters:
    max_iterations: int
    learning_rate: float
    exploration_prob: float
    optimism: float
    discount_factor: float
    convergence_methods: list[ConvergenceMethod] | None

    @staticmethod
    def combinations(convergence_methods: list[ConvergenceMethod] | None = None):
        combinations = []
        for max_iterations in [10000, 50000, 100000]:
            for exploration_prob in [0.1, 0.2, 0.3]:
                for learning_rate in [0.85, 0.9, 0.95]:
                    for optimism in [1, 2, 4]:
                        for discount_factor in [0.995, 0.95, 0.9]:
                            param_convergence_methods = (
                                [None]
                                if convergence_methods is None
                                else [None, convergence_methods]
                            )
                            for convergence_methods_p in param_convergence_methods:
                                combinations.append(
                                    AgentParameters(
                                        max_iterations,
                                        learning_rate,
                                        exploration_prob,
                                        optimism,
                                        discount_factor,
                                        convergence_methods_p,
                                    )
                                )

    def __str__(self):
        return (
            f"At most {padded_number(self.max_iterations)} iterations to "
            + padded_str(
                "completion"
                if self.convergence_methods is None
                else ",".join(self.convergence_methods),
                18,
            )
            + f"  Learning Rate: {padded_number(self.learning_rate)}"
            + f"  Exploration: {percent(self.exploration_prob)}"
            + f"  Optimism: {padded_number(self.optimism)}"
            + f"  Discount Factor: {percent(self.discount_factor)}"
        )

    def __lt__(self, other: "AgentParameters"):
        return (
            self.max_iterations,
            self.learning_rate,
            self.exploration_prob,
            self.optimism,
            self.discount_factor,
            self.convergence_methods or [],
        ) < (
            other.max_iterations,
            other.learning_rate,
            other.exploration_prob,
            other.optimism,
            other.discount_factor,
            other.convergence_methods or [],
        )

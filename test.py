from dataclasses import dataclass

import numpy as np
import pytest

from qlearn import (
    Agent,
    AgentPerformance,
    EvaluationMethod,
    create_spec,
    create_world,
    train_agent,
)


def padded_number(value: float | int, precision: int = 2, length: int = 6):
    num = str(round(value, precision))
    if len(num) < length:
        num = " " * (length - len(num)) + num
    return num


def percent(value: float):
    return padded_number(value * 100) + "%"


@dataclass
class TestResult:
    # identifiers
    world_size: int
    random_world: bool
    slippery: bool
    success_rate: float
    reward_schedule: tuple[float, float, float]
    evaluation_methods: list[EvaluationMethod] | None
    max_iterations: int

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
            f"Success: {percent(self.success_count / self.samples)}  "
            + f"Out of Moves: {percent(self.out_of_moves_count / self.samples)}  "
            + f"Unlucky Count: {percent(self.unlucky_count / self.samples)}  "
            + f"Average Total Moves: {padded_number(self.total_moves / self.samples)}"
            + f"Average Failed Moves: {padded_number(self.failed_moves / self.samples)}"
            + f"Average Failure Rate: {percent(self.failed_moves / self.total_moves)}"
        )

    def __lt__(self, other: "TestResult"):
        return (
            self.world_size,
            self.random_world,
            self.slippery,
            self.success_rate,
            self.reward_schedule,
            self.evaluation_methods,
            self.max_iterations,
        ) < (
            other.world_size,
            other.random_world,
            other.slippery,
            other.success_rate,
            other.reward_schedule,
            other.evaluation_methods,
            other.max_iterations,
        )


AGENT_RESULTS = []


@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize(
    "random_world,evaluation_methods",
    [
        (True, ["policy_convergence"]),
        (True, ["test_success"]),
        (True, None),
        (False, ["q_convergence"]),
        (False, None),
    ],
)
@pytest.mark.parametrize(
    "slippery,success_rate", [(True, 0.95), (True, 0.85), (True, 0.75), (False, 1)]
)
@pytest.mark.parametrize(
    "reward_schedule", [(10, -10, -0.001), (1, -1, -0.001), (10, -10, 0), (1, -1, 0)]
)
@pytest.mark.parametrize("max_iterations", [1000, 10000, 50000, 100000])
def test_agents(
    # world parameters
    world_size: int,
    random_world: bool,
    slippery: bool,
    success_rate: float,
    reward_schedule: tuple[float, float, float],
    # agent parameters
    evaluation_methods: list[EvaluationMethod] | None,
    max_iterations: int,
):
    spec = create_spec(world_size, random_world)

    def world_generator():
        return create_world(spec, slippery, success_rate, reward_schedule)

    result = TestResult(
        world_size,
        random_world,
        slippery,
        success_rate,
        reward_schedule,
        evaluation_methods,
        max_iterations,
    )
    agent = Agent(world_generator(), evaluation_methods)
    learned = agent.learn(max_iterations)
    if random_world:
        # large sample performance test
        for i in range(2000):
            agent.reset_world(world=world_generator())
            with agent.world:
                performance = agent.execute()
            result.accumulate(performance)
        print(result)

    else:
        # small sample equality test
        pass

    result.append


@pytest.fixture(scope="session", autouse=True)
def print_results():
    pass

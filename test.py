from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from tqdm import tqdm

from qlearn import (
    Agent,
    create_spec,
    create_world,
)
from utils import (
    AgentBatchPerformance,
    AgentParameters,
    AgentPerformance,
    ConvergenceMethod,
    WorldParameters,
    padded_number,
    percent,
)


@dataclass
class TestResult:
    world_parameters: WorldParameters
    agent_parameters: AgentParameters

    convergences: list[Literal[False] | int] = field(default_factory=list)
    batch_performance: AgentBatchPerformance = field(
        default_factory=AgentBatchPerformance
    )

    def train_accumulate(self, convergence: Literal[False] | int):
        self.convergences.append(convergence)

    def test_accumulate(self, performance: AgentPerformance):
        self.batch_performance.accumulate(performance)

    def converged(self):
        return any(self.convergences)

    def convergence_stats(self):
        if not self.converged():
            return (0, 0, 0)
        converged = [convergence for convergence in self.convergences if convergence]
        success_percent = len(converged) / len(self.convergences)
        success_average = float(np.average(converged))
        success_deviation = float(np.std(converged))
        return (success_percent, success_average, success_deviation)

    def __str__(self):
        success_percent, success_average, success_deviation = self.convergence_stats()
        return (
            str(self.world_parameters)
            + "\n"
            + str(self.agent_parameters)
            + "\n"
            + f"Converged {percent(success_percent)} of the time"
            + f" in {padded_number(success_average)} iterations on average"
            + f" (deviation of {padded_number(success_deviation)})"
            + "\n"
            + str(self.batch_performance)
        )


def test_agent(
    world_parameters: WorldParameters,
    agent_parameters: AgentParameters,
    train_iterations: int = 2,
    stochastic_test_iterations: int = 500,
):
    spec = create_spec(world_parameters.world_size, world_parameters.random_world)

    def world_generator():
        return create_world(
            spec,
            world_parameters.slippery,
            world_parameters.success_rate,
            world_parameters.reward_schedule,
        )

    agent = Agent(
        world_generator(),
        agent_parameters.convergence_method,
        learning_rate=agent_parameters.learning_rate,
        exploration_prob=agent_parameters.exploration_prob,
        optimism=agent_parameters.optimism,
        discount_factor=agent_parameters.discount_factor,
    )

    # train and test agent, averaging across all training and testing iterations
    result: TestResult = TestResult(world_parameters, agent_parameters)
    for _ in range(train_iterations):
        with agent.world:
            agent.reset_world(world=world_generator())
            agent.reset_internals()
            converged = agent.learn(agent_parameters.max_iterations)

        result.train_accumulate(converged)

        # performance test
        for _ in range(stochastic_test_iterations if world_parameters.slippery else 1):
            with agent.world:
                agent.reset_world(world=world_generator())
                performance = agent.execute()
            result.test_accumulate(performance)

    return result


def compute_stats():
    results: list[TestResult] = []
    for world_params in tqdm(
        WorldParameters.combinations(random_worlds=[True]), desc="Computing stats"
    ):
        result = test_agent(
            world_params,
            AgentParameters(
                discount_factor=0.9 if world_params.slippery else 1,
            ),
            train_iterations=10,
            stochastic_test_iterations=1000,
        )
        results.append(result)

    print()
    print("=" * 20)
    print("   Test Concluded   ")
    print("=" * 20)

    for test in sorted(results, key=lambda a: a.world_parameters):
        print("\n", test)


def hyperparameter_tuning():
    best_agents: list[TestResult] = []
    for world_params in tqdm(
        WorldParameters.combinations(
            # randomness doesn't help parameter tuning
            random_worlds=[False],
            # uncomment to pin values for parameter tuning
            # success_rates=[0.75],
            # reward_schedules=[(10, -10, -0.001)],
        ),
        desc="Testing worlds",
    ):
        best_agent: TestResult | None = None
        for agent_params in tqdm(
            AgentParameters.combinations(
                # uncomment to pin values for parameter tuning
                # max_iterations_l=[20000],
                # learning_rates=[0.3],
                # exploration_probs=[0.3],
                # optimisms=[1],
                # discount_factors=[0.9 if world_params.slippery else 1],
                convergence_methods=[ConvergenceMethod("policy_convergence")],
            ),
            desc=str(world_params),
        ):
            result = test_agent(
                world_params,
                agent_params,
                train_iterations=5,
                stochastic_test_iterations=1000,
            )
            # get the best converging agent (or non-converging if no agent converged)
            if best_agent is None or (
                best_agent.batch_performance < result.batch_performance
                and (result.converged() or not best_agent.converged())
            ):
                best_agent = result
        if best_agent is not None:
            best_agents.append(best_agent)

    print()
    print("=" * 20)
    print("   Test Concluded   ")
    print("=" * 20)

    for test in sorted(best_agents):
        print()
        print(test)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--hyperparameter-tuning",
        action="store_true",
        help="Execute with a large number of varied hyperparameters. Get a coffee, this will take a while",
    )
    args = parser.parse_args()

    if args.hyperparameter_tuning:
        hyperparameter_tuning()
    else:
        compute_stats()

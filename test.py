from dataclasses import dataclass, field

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
)


@dataclass
class TestResult:
    world_parameters: WorldParameters
    agent_parameters: AgentParameters

    batch_performance: AgentBatchPerformance = field(
        default_factory=AgentBatchPerformance
    )

    def accumulate(self, performance: AgentPerformance):
        self.batch_performance.accumulate(performance)

    def __str__(self):
        return (
            str(self.agent_parameters)
            + "\n"
            + str(self.world_parameters)
            + "\n"
            + str(self.batch_performance)
        )


AGENT_RESULTS = []


def test_agent(
    # world parameters
    world_size: int,
    random_world: bool,
    slippery: bool,
    success_rate: float,
    reward_schedule: tuple[float, float, float],
    # agent parameters
    evaluation_methods: list[ConvergenceMethod] | None,
    max_iterations: int,
):
    spec = create_spec(world_size, random_world)

    def world_generator():
        return create_world(spec, slippery, success_rate, reward_schedule)

    result = TestResult(
        WorldParameters(
            world_size=world_size,
            random_world=random_world,
            slippery=slippery,
            success_rate=success_rate,
            reward_schedule=reward_schedule,
        ),
        AgentParameters(
            evaluation_methods=evaluation_methods,
            max_iterations=max_iterations,
        ),
    )
    agent = Agent(world_generator(), evaluation_methods)
    learned = agent.learn(max_iterations)
    if random_world:
        # large sample performance test
        for i in range(500):
            agent.reset_world(world=world_generator())
            with agent.world:
                performance = agent.execute()
            result.accumulate(performance)

    else:
        # small sample equality test
        policy = None
        for i in range(2):
            agent.reset_world(world=world_generator())
            with agent.world:
                performance = agent.execute()
            result.accumulate(performance)
            if policy is not None:
                assert np.all(policy == agent.get_policy())
            policy = agent.get_policy()
        pass

    AGENT_RESULTS.append(result)


def print_results():
    yield
    print()
    print(i.samples for i in AGENT_RESULTS)
    for result in sorted(AGENT_RESULTS):
        print(result.samples)
        print(result)


def main():
    world_size_l = [4, 8]
    random_world_l = [True, False]
    success_rate_l = [0.75, 0.85, 0.95, 1]
    evaluation_method_l = [None, "some"]
    reward_schedule_l = [(10, -10, -0.001)]
    max_iterations_l = [100, 10000, 50000]

    combinations = int(
        np.prod(
            [
                len(param)
                for param in [
                    world_size_l,
                    random_world_l,
                    success_rate_l,
                    evaluation_method_l,
                    reward_schedule_l,
                    max_iterations_l,
                ]
            ]
        )
    )

    with tqdm(total=combinations) as pbar:
        for world_size in world_size_l:
            for random_world in random_world_l:
                for success_rate in success_rate_l:
                    slippery = success_rate < 1
                    evaluation_method: ConvergenceMethod = (
                        "policy_convergence" if slippery else "q_convergence"
                    )
                    for evaluation_methods in [None, [evaluation_method]]:
                        for reward_schedule in reward_schedule_l:
                            for max_iterations in max_iterations_l:
                                test_agent(
                                    world_size,
                                    random_world,
                                    slippery,
                                    success_rate,
                                    reward_schedule,
                                    evaluation_methods,
                                    max_iterations,
                                )
                                pbar.update(1)

    print_results()


if __name__ == "__main__":
    main()

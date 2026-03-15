import os
from argparse import ArgumentParser
from shutil import rmtree

from termcolor import cprint

from game import Game
from informed import AStarSearch, IDAStarSearch
from replay import Replay
from solver import Solver
from uninformed import BreadthFirstSearch, GamblersSearch

all_states = sorted(
    [
        os.path.basename(file).split(".")[0]
        for file in os.listdir("states")
        if os.path.isfile(os.path.join("states", file))
    ]
)
all_solvers = {
    "gamblers": GamblersSearch,
    "bfs": BreadthFirstSearch,
    "idastar": IDAStarSearch,
    "astar": AStarSearch,
}


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "solvers",
        nargs="+",
        choices=all_solvers.keys(),
        help="List of solvers to use",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all old solution files for SOLVERS before solving",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=all_states,
        help="List of names (without extension) of problem states to solve (located in states/)",
    )
    parser.add_argument(
        "--random",
        type=int,
        help="Solve a random state with n_bolts=RANDOM. Overrides --states",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay solutions from STATES and SOLVERS. Cannot be used with --random",
    )
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save_solution",
        help="Do not save the solution. Solutions are saved by default",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of iterations to run. Solvers not complete within MAX_ITERATIONS will be considered failed",
    )
    parser.add_argument(
        "--print-steps",
        action="store_true",
        help="Print the state after each step. Recommended to use with --step-delay-ms for visualization",
    )
    parser.add_argument(
        "--step-delay-ms",
        type=int,
        default=0,
        help="Delay between steps in milliseconds",
    )

    args = parser.parse_args()

    solver_instances = [
        all_solvers[solver_name](
            print_steps=args.print_steps, step_delay_ms=args.step_delay_ms
        )
        for solver_name in args.solvers
    ]

    if args.clean:
        for solver in solver_instances:
            rmtree(os.path.join("solutions", solver.get_name()), ignore_errors=True)

    if args.random:
        if args.replay:
            raise RuntimeError("Cannot replay a random game")
        solve_random(
            solver_instances,
            args.random,
            save_solution=args.save_solution,
            max_iterations=args.max_iterations,
        )
    elif args.replay:
        replay_solutions(
            solver_instances,
            args.states,
            save_solution=args.save_solution,
            max_iterations=args.max_iterations,
        )
    else:
        for state in args.states:
            solve(
                solver_instances,
                Game.from_state_file(state),
                save_solution=args.save_solution,
                max_iterations=args.max_iterations,
            )


def solve(
    solvers: list[Solver], game: Game, save_solution: bool, max_iterations: int | None
) -> None:
    state = game.snapshot()
    for solver in solvers:
        solver.solve(game, save_solution=save_solution, max_iterations=max_iterations)
        game.restore_snapshot(state)
        print("\n")


def solve_random(
    solvers: list[Solver], size: int, save_solution: bool, max_iterations: int | None
) -> None:
    solve(
        solvers,
        Game.random_state(size),
        save_solution=save_solution,
        max_iterations=max_iterations,
    )


def replay_solutions(
    solvers: list[Solver],
    states: list[str],
    save_solution: bool,
    max_iterations: int | None,
):
    for solver in solvers:
        solver_name = solver.get_name()
        solution_dir = os.path.join("solutions", solver_name)
        all_solutions = [
            solution
            for solution in os.listdir(solution_dir)
            if os.path.isfile(os.path.join(solution_dir, solution))
        ]
        for state in states:
            solutions = [
                solution for solution in all_solutions if solution.startswith(state)
            ]
            if len(solutions) == 0:
                cprint(f"No solutions found for state {solver_name}/{state}", "red")
                continue
            for solution_name in solutions:
                solve(
                    [Replay(solver_name=solver_name, solution_name=solution_name)],
                    Game.from_state_file(state),
                    save_solution=save_solution,
                    max_iterations=max_iterations,
                )


if __name__ == "__main__":
    main()

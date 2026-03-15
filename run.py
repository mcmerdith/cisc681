import os
from argparse import ArgumentParser
from shutil import rmtree

from game import Game
from informed import AStarSearch, IDAStarSearch
from uninformed import BreadthFirstSearch

states = ["problem1", "problem2", "problem3"]
solvers = {
    "bfs": BreadthFirstSearch,
    "idastar": IDAStarSearch,
    "astar": AStarSearch,
}


if __name__ == "__main__":
    fn = ArgumentParser()
    fn.add_argument(
        "--clean",
        action="store_true",
        help="Remove all old solution files before solving",
    )
    fn.add_argument(
        "--states", nargs="+", default=states, help="List of problem states to solve"
    )
    fn.add_argument(
        "--solvers", nargs="+", default=solvers.keys(), help="List of solvers to use"
    )
    fn.add_argument("--random", type=int, help="Solve a random state with RANDOM bolts")
    fn.add_argument(
        "--print-state", action="store_true", help="Print the state after each step"
    )
    fn.add_argument(
        "--no-save",
        action="store_false",
        dest="save_solution",
        help="Do not save the solution",
    )

    args = fn.parse_args()

    if args.clean:
        for solver_name in args.solvers:
            solver = solvers[solver_name]()
            rmtree(os.path.join("solutions", solver.get_name()), ignore_errors=True)
        exit()

    game = None
    if args.random:
        game = Game.random_state(args.random)
        args.states = [game.snapshot()]

    for state in args.states:
        for solver_name in args.solvers:
            solver = solvers[solver_name](print_steps=args.print_state)
            if args.clean:
                rmtree(os.path.join("solutions", solver.get_name()), ignore_errors=True)
            if isinstance(state, str):
                game = Game.from_state_file(state)
            elif game:
                game.restore_snapshot(state)
            else:
                raise ValueError("Game not initialized")
            solver.solve(game, save_solution=args.save_solution)

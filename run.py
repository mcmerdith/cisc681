import os

from solver import Solver

if __name__ == "__main__":
    from argparse import ArgumentParser

    from game import Game
    from informed import AStarSearch
    from uninformed import UniformCostSearch

    states = ["problem1", "problem2", "problem3"]
    solvers = {"astar": AStarSearch, "uniform_cost": UniformCostSearch}

    fn = ArgumentParser()
    fn.add_argument("--clean", action="store_true")
    fn.add_argument("--states", nargs="+", default=states)
    fn.add_argument("--solvers", nargs="+", default=solvers.keys())
    fn.add_argument("--random", type=int)
    fn.add_argument("--print-state", action="store_true")
    fn.add_argument("--no-save", action="store_false", dest="save_solution")

    args = fn.parse_args()

    if args.random:
        args.states = ["random"]

    for state in args.states:
        for solver_name in args.solvers:
            if args.clean:
                os.rmdir(os.path.join("solutions", state))
            if state == "random":
                game = Game.random_state(args.random)
            else:
                game = Game.from_state_file(state)
            solver = solvers[solver_name](print_steps=args.print_state)
            solver.solve(game, save_solution=args.save_solution)

import argparse
from random import randint

from game import Game, GameState
from solver import Solver


class GamblersSearch(Solver):
    """
    The worst search algorithm. Let's roll some dice!
    """

    def iteration(self, game: Game):
        while True:
            # get random bolts to work on
            from_idx = randint(0, len(game.game_state.bolts) - 1)
            to_idx = randint(0, len(game.game_state.bolts) - 1)

            # skip moves on the same bolt
            if from_idx == to_idx:
                continue

            # repeat until a valid move is made
            moved = game.swap_nut(from_idx, to_idx)
            if moved:
                break

        game.game_state.cost += 4 / moved


class UniformCostSearch(Solver):
    """
    An implementation of the uniform cost search algorithm
    """

    expanded: dict[GameState, GameState] = {}
    """All states that have been previously expanded. Each state is mapped to itself"""

    fringe: dict[GameState, GameState] = {}
    """All states that are currently in the fringe. Each state is mapped to itself"""

    def get_stats(self):
        min_fringe = min(self.fringe.values(), key=lambda s: s.cost)
        max_fringe = max(self.fringe.values(), key=lambda s: s.cost)

        return "\n".join(
            [
                super().get_stats(),
                f"Expanded {len(self.expanded)} states",
                f"Fringe: {len(self.fringe)} states",
                f"  Min: {min_fringe.cost:.2f} ({len(min_fringe.actions)} steps)",
                f"  Max: {max_fringe.cost:.2f} ({len(max_fringe.actions)} steps)",
            ]
        )

    def best_fringe_node(self):
        """The lowest cost state currently on the fringe"""

        return min(self.fringe.values(), key=lambda s: s.cost)

    def expand(self, game: Game):
        """Expand the best fringe node and add its children to the fringe"""

        # restore the best fringe node as the current game state
        game.restore_snapshot(self.best_fringe_node())

        # if this state is solved, do nothing and let control fall back to the base solver
        if game.solved():
            return

        # remove this state from the fringe and add it to the expanded set
        self.fringe.pop(game.game_state)
        self.expanded[game.game_state] = game.snapshot()

        # compute all possible moves from the current state
        for from_idx in range(0, len(game.game_state.bolts)):
            for to_idx in range(0, len(game.game_state.bolts)):
                # ignore useless moves
                if from_idx == to_idx:
                    continue

                # perform a dry-run to calculate the validity of the move
                moved = game.swap_nut(from_idx, to_idx, dry_run=True)

                # ignore failed moves
                if moved == 0:
                    continue

                # success, update cost and take a snapshot
                game.next_game_state.cost += 4 / moved
                snapshot = game.snapshot(next_state=True)

                # prune nodes that don't improve on something already expanded
                if snapshot in self.expanded:
                    if self.expanded[snapshot].cost < snapshot.cost:
                        continue

                # replace nodes on the fringe with better ones, or prune them
                if snapshot in self.fringe:
                    if snapshot.cost < self.fringe[snapshot].cost:
                        self.fringe.pop(snapshot)
                    else:
                        continue

                # add this state to the fringe
                self.fringe[snapshot] = snapshot

    def iteration(self, game: Game):
        # initial state
        if self._iteration == 0:
            snapshot = game.snapshot()
            self.fringe[snapshot] = snapshot

        # fail if we run out of states to expand somehow
        if len(self.fringe) == 0:
            raise RuntimeError("Unsolvable problem!")

        # continue to expand the fringe
        self.expand(game)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "state_file",
        help="The name of the game state file to load, excluding the extension",
    )
    args = parser.parse_args()
    game = Game.from_state(args.state_file)
    uniform_cost = UniformCostSearch()
    uniform_cost.solve(game, save_solution=True)


if __name__ == "__main__":
    main()

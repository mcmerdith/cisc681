import argparse
import heapq
from dataclasses import dataclass, field
from math import ceil, inf

from game import Game, GameState
from solver import Solver


def heuristic(state: GameState) -> float:
    return ceil(sum([bolt.boundaries() for bolt in state.bolts]) / 2)


@dataclass
class AStarSearch(Solver):
    """
    An implementation of the A* algorithm
    """

    expanded: dict[GameState, float] = field(default_factory=dict)
    """All states that have been previously expanded"""

    fringe: list[GameState] = field(default_factory=list)
    """All states that are currently in the fringe"""

    def get_stats(self):
        min_fringe = min(self.fringe)
        max_fringe = max(self.fringe)

        return "\n".join(
            [
                super().get_stats(),
                f"Expanded {len(self.expanded)} states",
                f"Fringe: {len(self.fringe)} states",
                f"  Min: {min_fringe.cost:.2f} + {min_fringe.heuristic:.2f} ({len(min_fringe.actions)} steps)",
                f"  Max: {max_fringe.cost:.2f} + {max_fringe.heuristic:.2f} ({len(max_fringe.actions)} steps)",
            ]
        )

    def is_expanded(self, state: GameState) -> bool:
        return state in self.expanded and self.expanded[state] <= state.cost

    def pop_best_state(self) -> GameState:
        while True:
            # restore the best fringe node as the current game state
            try:
                best_state = heapq.heappop(self.fringe)
            except IndexError:
                raise RuntimeError("Unsolvable problem!")

            # select a node that has either not been expanded or has a lower cost than what was expanded
            if not self.is_expanded(best_state):
                break
        return best_state

    def expand(self, game: Game):
        """Expand the best fringe node and add its children to the fringe"""

        best_state = self.pop_best_state()
        self.expanded[best_state] = best_state.cost

        # if this state is solved, do nothing and let control fall back to the base solver
        if best_state.solved():
            game.restore_snapshot(best_state)
            return

        changed = True
        # compute all possible moves from the current state
        for from_idx in range(0, len(game.game_state.bolts)):
            for to_idx in range(0, len(game.game_state.bolts)):
                if changed:
                    # optimization to reduce the number of calls to `deepcopy`
                    game.restore_snapshot(best_state)
                    changed = False

                # perform a dry-run to calculate the validity of the move
                moved = game.swap_nuts(from_idx, to_idx)

                # ignore failed moves
                if moved == 0:
                    continue

                changed = True

                # skip already expanded states
                if self.is_expanded(game.game_state):
                    continue

                # increase state cost based on the number of nuts moved
                game.game_state.cost += 1
                # predict the heuristic cost based on the number of mismatched groups
                game.game_state.heuristic = heuristic(game.game_state)

                # add this state to the fringe
                snapshot = game.snapshot()
                heapq.heappush(self.fringe, snapshot)

    def iteration(self, game: Game):
        # initial state
        if self._iteration == 0:
            snapshot = game.snapshot()
            self.fringe = [snapshot]

        # continue to expand the fringe
        self.expand(game)


@dataclass
class IDAStarSearch(Solver):
    """
    An implementation of the IDA* search algorithm
    """

    threshold: float = field(default=0.0)
    """The threshold for the next iteration"""

    expanded: int = field(default=0)
    """Count of states that have been expanded"""

    def get_stats(self):
        return "\n".join(
            [
                super().get_stats(),
                f"Expanded {self.expanded} states",
                f"Threshold: {self.threshold}",
            ]
        )

    def search(
        self, game: Game, state: GameState, parents: list[GameState] | None = None
    ):
        """Expand the best fringe node and add its children to the fringe"""

        total_cost = state.heuristic_cost()
        if self.threshold < total_cost:
            return total_cost

        if state.solved():
            game.restore_snapshot(state)
            return True

        self.expanded += 1
        min_excess = inf

        if not parents:
            parents = []

        changed = True
        # compute all possible moves from the current state
        for from_idx in range(0, len(game.game_state.bolts)):
            for to_idx in range(0, len(game.game_state.bolts)):
                if changed:
                    # optimization to reduce the number of calls to `deepcopy`
                    game.restore_snapshot(state)
                    changed = False

                # perform a dry-run to calculate the validity of the move
                moved = game.swap_nuts(from_idx, to_idx)

                # ignore failed moves
                if moved == 0:
                    continue

                changed = True

                if game.game_state in parents:
                    # avoid branching to states we're already visited in this pass
                    continue

                # increase state cost based on the number of nuts moved
                game.game_state.cost += 1
                # predict the heuristic cost based on the number of mismatched groups
                game.game_state.heuristic = heuristic(game.game_state)

                # continue deepening the search
                result = self.search(game, game.game_state, parents + [game.game_state])
                if result is True:
                    return True
                if result < min_excess:
                    min_excess = result

        return min_excess

    def iteration(self, game: Game):
        # initial state
        if self._iteration == 0:
            snapshot = game.snapshot()
            snapshot.heuristic = heuristic(snapshot)
            self.threshold = snapshot.heuristic

        snapshot = game.snapshot()
        result = self.search(game, snapshot)

        if result is True:
            return
        elif result == inf:
            raise RuntimeError("Unsolvable problem!")
        else:
            self.threshold = result
            game.restore_snapshot(snapshot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "state_file",
        help="The name of the game state file to load, excluding the extension",
    )
    parser.add_argument(
        "--print-state",
        action="store_true",
        help="Print the state after each move",
    )
    parser.add_argument(
        "--no-save",
        action="store_false",
        help="Do not save the solution",
        dest="save_solution",
    )
    args = parser.parse_args()
    game = Game.from_state_file(args.state_file)
    astar = AStarSearch(print_steps=args.print_state)
    astar.solve(game, save_solution=args.save_solution)


if __name__ == "__main__":
    main()

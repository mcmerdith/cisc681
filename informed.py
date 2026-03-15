import argparse
import heapq
from dataclasses import dataclass, field
from math import inf

from game import Game, GameState
from solver import Solver


def heuristic(state: GameState) -> float:
    bolt_colors = [set(bolt._slots).difference({0}) for bolt in state.bolts]
    all_colors = set().union(*bolt_colors)

    # needed to determine how many bolts are required for each color
    nut_counts = {
        color: sum(1 for bolt in state.bolts for slot in bolt._slots if slot == color)
        for color in all_colors
    }

    return sum(
        [
            max(
                0,
                sum(1 for bolt in bolt_colors if color in bolt) - nut_counts[color] / 4,
            )
            for color in all_colors
        ]
    )


@dataclass
class AStarSearch(Solver):
    """
    An implementation of the A* algorithm
    """

    expanded: dict[GameState, float] = field(default_factory=dict)
    """All states that have been previously expanded"""

    fringe: list[GameState] = field(default_factory=list)
    """All states that are currently in the fringe"""

    _max_fringe_size: int = field(default=0, init=False)

    def reset(self):
        super().reset()
        self.expanded = dict()
        self.fringe = list()
        self._max_fringe_size = 0

    def get_stats(self):

        return "\n".join(
            [
                super().get_stats(),
                f"Expanded {len(self.expanded)} states",
                f"Fringe size: {len(self.fringe)} states",
                f"Max fringe size: {self._max_fringe_size} states",
            ]
        )

    def is_expanded(self, state: GameState) -> bool:
        return state in self.expanded and self.expanded[state] <= state.cost

    def pop_best_state(self) -> GameState | None:
        current_size = len(self.fringe)
        if self._max_fringe_size < current_size:
            self._max_fringe_size = current_size

        while True:
            # restore the best fringe node as the current game state
            try:
                best_state = heapq.heappop(self.fringe)
            except IndexError:
                return None

            # select a node that has either not been expanded or has a lower cost than what was expanded
            if not self.is_expanded(best_state):
                break
        return best_state

    def iteration(self, game: Game) -> bool:
        # initial state
        if self._iteration == 0:
            snapshot = game.snapshot()
            self.fringe = [snapshot]

        # continue to expand the fringe
        best_state = self.pop_best_state()
        if not best_state:
            return True
        self.expanded[best_state] = best_state.cost

        # if this state is solved, do nothing and let control fall back to the base solver
        if best_state.solved():
            game.restore_snapshot(best_state)
            return True

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

        return False


@dataclass
class IDAStarSearch(Solver):
    """
    An implementation of the IDA* search algorithm
    """

    threshold: float = field(default=0.0)
    """The threshold for the next iteration"""

    expanded: int = field(default=0)
    """Count of states that have been expanded"""

    def reset(self):
        super().reset()
        self.expanded = 0
        self.threshold = 0

    def get_stats(self):
        return "\n".join(
            [
                super().get_stats(),
                f"Expanded {self.expanded} states",
                f"Threshold: {self.threshold}",
            ]
        )

    def search(
        self, game: Game, state: GameState, backtrack: list[GameState] | None = None
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

        if not backtrack:
            backtrack = []

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

                if game.game_state in backtrack:
                    # avoid branching to states we're already visited in this pass
                    continue

                # increase state cost based on the number of nuts moved
                game.game_state.cost += 1
                # predict the heuristic cost based on the number of mismatched groups
                game.game_state.heuristic = heuristic(game.game_state)

                # continue deepening the search
                result = self.search(
                    game, game.game_state, backtrack + [game.game_state]
                )
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

        if result is True or result == inf:
            return True
        else:
            self.threshold = result
            game.restore_snapshot(snapshot)
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "state_file",
        help="The name of the game state file to load, excluding the extension",
    )
    args = parser.parse_args()
    IDAStarSearch().solve(Game.from_state_file(args.state_file), save_solution=True)


if __name__ == "__main__":
    main()

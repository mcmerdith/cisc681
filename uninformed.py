import argparse
from collections import deque
from dataclasses import dataclass, field
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
            moved = game.swap_nuts(from_idx, to_idx)
            if moved:
                break

        game.game_state.cost += 4 / moved
        return False


@dataclass
class BreadthFirstSearch(Solver):
    """
    An implementation of the uniform cost search algorithm
    """

    visited: set[GameState] = field(default_factory=set)
    """All states that have been previously expanded"""

    fringe: deque[GameState] = field(default_factory=deque)
    """All states that are currently in the fringe"""

    def get_stats(self):
        return "\n".join(
            [
                super().get_stats(),
                f"Expanded {len(self.visited)} states",
                f"Fringe: {len(self.fringe)} states",
            ]
        )

    def pop_best_state(self) -> GameState | None:
        while True:
            # restore the best fringe node as the current game state
            try:
                best_state = self.fringe.popleft()
            except IndexError:
                return None

            # select a node that has not been expanded
            if best_state not in self.visited:
                break
        return best_state

    def iteration(self, game: Game) -> bool:
        # initial state
        if self._iteration == 0:
            snapshot = game.snapshot()
            self.fringe = deque([snapshot])

        # continue to expand the fringe
        best_state = self.pop_best_state()
        if not best_state:
            return True
        self.visited.add(best_state)

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

                moved = game.swap_nuts(from_idx, to_idx)

                # ignore failed moves
                if moved == 0:
                    continue

                changed = True

                # skip already expanded states
                if game.game_state in self.visited:
                    continue

                # not actually used other than for stats
                game.game_state.cost += 1

                # add this state to the fringe
                snapshot = game.snapshot()
                self.fringe.append(snapshot)

        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "state_file",
        help="The name of the game state file to load, excluding the extension",
    )
    args = parser.parse_args()
    BreadthFirstSearch().solve(
        Game.from_state_file(args.state_file), save_solution=True
    )


if __name__ == "__main__":
    main()

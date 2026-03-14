import argparse
import heapq
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


class UniformCostSearch(Solver):
    """
    An implementation of the uniform cost search algorithm
    """

    expanded: dict[GameState, float] = {}
    """All states that have been previously expanded"""

    fringe: list[GameState] = []
    """All states that are currently in the fringe"""

    def get_stats(self):
        min_fringe = min(self.fringe)
        max_fringe = max(self.fringe)

        return "\n".join(
            [
                super().get_stats(),
                f"Expanded {len(self.expanded)} states",
                f"Fringe: {len(self.fringe)} states",
                f"  Min: {min_fringe.cost:.2f} ({len(min_fringe.actions)} steps)",
                f"  Max: {max_fringe.cost:.2f} ({len(max_fringe.actions)} steps)",
            ]
        )

    def is_expanded(self, state: GameState) -> bool:
        return state in self.expanded and self.expanded[state] < state.cost

    def pop_best_state(self) -> GameState:
        while True:
            # restore the best fringe node as the current game state
            best_state = heapq.heappop(self.fringe)

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

        changed = False
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

                # skip already expanded states
                if self.is_expanded(game.game_state):
                    continue

                changed = True

                # increase state cost based on the number of nuts moved
                game.game_state.cost += 4 / moved

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "state_file",
        help="The name of the game state file to load, excluding the extension",
    )
    args = parser.parse_args()
    game = Game.from_state(args.state_file)
    uniform_cost = UniformCostSearch()
    uniform_cost.solve(game, save_solution=False)


if __name__ == "__main__":
    main()

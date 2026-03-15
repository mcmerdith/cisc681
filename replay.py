import os
from dataclasses import dataclass, field

from game import Game
from solver import Solver


@dataclass
class Replay(Solver):
    solver_name: str | None = None
    solution_name: str | None = None

    _solution_path: str = field(init=False)
    _solution: list[tuple[int, int]] = field(init=False)

    def __post_init__(self):
        if self.solver_name is None:
            raise ValueError("solver_name must be provided")
        if self.solution_name is None:
            raise ValueError("solution_name must be provided")

        self.print_steps = True
        self.step_delay_ms = 1000

        self._solution_path = os.path.join("solutions", self.solver_name, self.solution_name)
        if not os.path.exists(self._solution_path):
            raise ValueError(f"{self._solution_path} does not exist")

        with open(self._solution_path, "r") as f:
            self._moves = [
                tuple(int(idx) for idx in line.strip().split())
                for line in f.readlines()
            ]
        for move in self._moves:
            if len(move) != 2:
                raise ValueError(
                    f"Invalid move: {move} (expected tuple[bolt_from, bolt_to])"
                )

    def get_name(self) -> str:
        return f"Replay({self._solution_path})"

    def iteration(self, game: Game) -> bool:
        if self._iteration == 0:
            if len(self._moves) == 0:
                if game.solved():
                    return True
                else:
                    raise ValueError("Solution is empty")

        if len(self._moves) <= self._iteration:
            return True

        move = self._moves[self._iteration]
        game.swap_nuts(move[0] - 1, move[1] - 1)
        game.game_state.cost += 1

        return False

    def solve(
        self,
        game: Game,
        max_iterations: int | None = None,
        save_solution: bool = False,
    ) -> None:
        return super().solve(game, None, False)

    def _save_solution(self, game: Game) -> None:
        pass

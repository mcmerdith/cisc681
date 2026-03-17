from blessed import Terminal

from game import Game
from solver import Solver


class Interactive(Solver):
    _from: int | None = None
    _to: int | None = None
    _term: Terminal
    _term_width: int
    _term_height: int

    def __post_init__(self):
        self.print_steps = False
        self.step_delay_ms = 1000

    def reset(self):
        super().reset()
        self._from = None
        self._to = None

    def input_move(self, game: Game) -> tuple[int, int] | None:
        blank_line = " " * (len(game.game_state.bolts) * 2 - 1)

        idx = self._to or 0
        self._to = self._from = None

        with self._term.cbreak(), self._term.hidden_cursor(), self._term.fullscreen():
            y, x = (
                self._term_height // 2 - 3,
                self._term_width // 2 - len(blank_line) // 2,
            )

            while True:
                print(self._term.home + self._term.clear)
                for i, line in enumerate(game.colored_string().split("\n")):
                    with self._term.location(x, y + i):
                        print(line)
                with self._term.location(x, y):
                    print(blank_line)
                if self._from is None:
                    with self._term.location(x + idx * 2, y):
                        print("^")
                else:
                    with self._term.location(x, y):
                        print(game.build_action_indicator((self._from, idx)))

                key = self._term.inkey(0.1)

                if key.name == "KEY_LEFT" or key == "a":
                    idx = max(0, idx - 1)
                elif key.name == "KEY_RIGHT" or key == "d":
                    idx = min(len(game.game_state.bolts) - 1, idx + 1)
                elif key.name == "KEY_ENTER" or key == " ":
                    if self._from is not None:
                        self._to = idx
                        return (self._from, self._to)
                    else:
                        self._from = idx
                        self._to = None
                elif key.name == "KEY_ESCAPE":
                    if self._from is not None:
                        self._from = None
                        self._to = None
                    else:
                        return None
                elif key.is_backspace(None, True):
                    self._from = None
                    self._to = None

    def iteration(self, game: Game):
        while True:
            move = self.input_move(game)
            if move is None:
                break
            if game.swap_nuts(*move):
                game.game_state.cost += 1
                break

        return move is None

    def solve(
        self,
        *p,
        **kw,
    ) -> None:
        self._term = Terminal()
        self._term_width = self._term.width
        self._term_height = self._term.height
        super().solve(*p, **kw)


if __name__ == "__main__":
    game = Game.random_state(4)
    Interactive().solve(game)

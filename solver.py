from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from os import makedirs, path
from time import sleep, time

import colorama
from termcolor import cprint

from game import Game


@dataclass
class Solver(ABC):
    """
    Base class for all solvers.

    Must implement `reset` and `iteration`.

    May implement `get_stats`. Call `super().get_stats()` if overriding.
    """

    print_steps: bool = False
    """If a state visualization should be printed after each iteration"""

    step_delay_ms: int = 0
    """Delay between steps in milliseconds. Useful for visualizing the solver's actions"""

    _iteration: int = field(default=0, init=False)
    """The current iteration number"""

    _start_time: float | None = field(default=None, init=False)
    """The start time of the solver run"""

    _end_time: float | None = field(default=None, init=False)
    """The end time of the solver run"""

    _halt: bool = field(default=False, init=False)
    """Whether the solver should halt"""

    def reset(self):
        """Set the initial state of the solver"""

        # reset solver state
        self._iteration = 0
        self._start_time = time()
        self._end_time = None
        self._halt = False

    @abstractmethod
    def iteration(self, game: Game) -> bool:
        """
        Perform one iteration of the solver on the given game.

        Returns True if the solver should halt, False otherwise.

        Solvers are responsible for updating the cost of the iteration on each game state.

        Solvers should not call `iteration()` directly.
        """

        raise NotImplementedError("Solvers must implement iteration()")

    def get_stats(self) -> str:
        """
        Returns a string representation of the solver's stats, including the time taken.

        Override to add additional statistics.
        """

        return f"Time taken: {self.time_taken()}s"

    def get_name(self) -> str:
        """The name of the solver. Determined automatically, not necessary to override."""

        return type(self).__name__

    def time_taken(self) -> float | None:
        """
        Returns the time taken by the solver to solve the game, in seconds,
        or None if the solver either has not been started or has not finished.

        Rounded to 2 decimal places.
        """

        return (
            round(self._end_time - self._start_time, 2)
            if self._start_time and self._end_time
            else None
        )

    def solve(
        self,
        game: Game,
        max_iterations: int | None = None,
        save_solution: bool = False,
    ) -> None:
        """
        The main solver loop, calling `iteration()` on until the game is solved
        or `max_iterations` is reached.

        If `save_solution` is True, files containing the solution and statistics
        will be saved to `solutions/<solver_name>/<state_name>.txt
        """

        self.reset()

        # fix windows console color handling
        colorama.just_fix_windows_console()

        cprint(f"Solving {game.state_name} using {self.get_name()}...\n", "light_blue")

        while True:
            solved = game.solved()
            failed = (self._halt and not solved) or (
                max_iterations is not None and self._iteration >= max_iterations
            )

            # always print the first and last state, and steps when print_steps is True
            if self._iteration == 0 or self.print_steps or solved or failed:
                output = game.colored_string()

                if self._iteration > 0:
                    # overwrite the previous output
                    print(colorama.Cursor.UP(len(output.split("\n")) + 2))

                print(output, "\n")

            if self.step_delay_ms > 0:
                sleep(self.step_delay_ms / 1000.0)

            if solved or failed:
                break

            self._halt = self.iteration(game)
            self._iteration += 1

        self._end_time = time()

        if solved:
            cprint(f"{self.get_name()} solved in {self._iteration} iterations", "green")
            if save_solution:
                self._save_solution(game)
            print()
            cprint(f"{game.get_stats()}", "light_blue")
        else:
            cprint(f"Failed to solve within {max_iterations} iterations", "red")

        cprint(f"{self.get_stats()}", "light_blue")

    def _save_solution(self, game: Game):
        out_dir = path.join("solutions", self.get_name())
        makedirs(path.join(out_dir, "stats"), exist_ok=True)

        def make_path(name, *paths: str):
            return path.join(out_dir, *paths, name + ".txt")

        filenum = 0
        filename = f"{game.state_name}_{filenum}"

        while path.exists(make_path(filename)):
            filenum += 1
            filename = f"{game.state_name}_{filenum}"

        # write the solution file
        with open(make_path(filename), "w") as f:
            f.writelines(
                f"{action[0] + 1} {action[1] + 1}\n"
                for action in game.game_state.actions
            )

        cprint(f"Saved solution to {make_path(filename)}", "light_blue")

        # write the statistics file
        with open(make_path(filename, "stats"), "w") as f:
            f.write(f"Initial state\n{game.initial_state_str}\n\n")
            f.write(f"Solution\n{str(game)}\n\n")
            f.write(f"Solved in {self._iteration} iterations\n")
            f.write(f"{game.get_stats()}\n")
            f.write(f"{self.get_stats()}\n")

        cprint(f"Saved statistics to {make_path(filename, 'stats')}", "light_blue")

from copy import deepcopy
from dataclasses import dataclass, field
from functools import reduce
from typing import Callable

import numpy as np
from numpy.random import permutation
from termcolor import colored, cprint

COLORS = [
    "default",  # 0 represents no nut, so this should never be indexed
    "yellow",
    "red",
    "blue",
    "green",
    "magenta",
    "cyan",
    "light_yellow",
    "light_red",
    "light_green",
    "light_magenta",
]


def colored_nut_str(nut: int) -> str:
    """Get a colored terminal character for a nut"""
    if nut == 0:
        return "|"  # empty => bolt
    elif nut < len(COLORS):
        return colored("0", COLORS[nut])  # colored if we have it
    else:
        return str(nut)  # just the value otherwise


@dataclass
class Bolt:
    _slots: list[int] = field(default_factory=list)
    """The slots on the bolt, with 0 indicating an empty space"""

    def nuts(self) -> list[int]:
        """Get the indices of nuts on the bolt"""

        return [i for i, slot in enumerate(self._slots) if slot != 0]

    def free_slots(self) -> int:
        """Get the number of free slots on the bolt"""

        return self._slots.count(0)

    def can_accept_nuts(self, new_nuts: tuple[int, int]) -> bool:
        """Nuts can only be placed on top of matching colors, or if the bolt is empty"""

        count, color = new_nuts

        # can't have no nuts
        if count <= 0:
            return False

        # can't have more than 4 nuts
        if count > 4:
            return False

        # can't push into a non-empty slot
        if not self._slots[count - 1] == 0:
            return False

        nuts = self.nuts()

        # either there's no nuts, or the top one matches
        return len(nuts) == 0 or self._slots[nuts[0]] == color

    def solved(self) -> bool:
        """The bolt is solved if all nuts are the same color, or there are no nuts"""

        return len(set(self._slots)) <= 1

    def collapse_spaces(self) -> None:
        """Collapse any empty spaces between nuts and the bottom of the bolt"""

        self._slots = [slot for slot in self._slots if slot != 0]
        self._slots = [*[0] * (4 - len(self._slots)), *self._slots]

    def validate(self, initial=False) -> bool:
        """
        Verify that the current state is valid, printing an error message if not.

        If `initial` is True, a bolt must have either 0 or 4 nuts (full or empty).

        Returns False if the current state is not valid.
        """

        if len(self._slots) != 4:
            cprint(
                f"Invalid state: bolt has {len(self._slots)} slots, expected 4", "red"
            )
            return False

        if initial:
            nut_count = len(self.nuts())
            if nut_count != 0 and nut_count != 4:
                cprint(
                    f"Invalid state: bolt has {len(self.nuts())} nuts, expected 0 or 4",
                    "red",
                )
                return False

        def check_nuts(acc: list[int], slot: int):
            """Check that there are no spaces between nuts, replacing any illegal nut with -1"""
            if len(acc) == 0:
                return [slot]
            if slot < 0 or slot > 9:
                return acc + [-1]
            if slot == 0 and not all(s == 0 for s in acc):
                # empty spaces may only be preceeded by empty spaces
                return acc + [-1]
            return acc + [slot]

        valid_slots = reduce(check_nuts, self._slots, [])
        if any(slot < 0 for slot in valid_slots):
            cprint(
                f"Invalid state: bolt has an illegal slot at {[i for i, slot in enumerate(valid_slots) if slot < 0]}",
                "red",
            )
            return False

        return True


@dataclass(eq=False)
class GameState:
    """
    Represents the state of the game including the bolts,
    actions taken, and cost incurred
    """

    bolts: list[Bolt]
    """The bolts that make up the game state"""

    actions: list[tuple[int, int]] = field(default_factory=list, compare=False)
    """The actions taken to reach this state"""

    cost: float = field(default=0.0, compare=False)
    """The cost of reaching this state"""

    heuristic: float = field(default=0.0, compare=False)
    """The heuristic value of this state. If not using a heuristic this should be left at 0"""

    def last_action(self) -> tuple[int, int] | None:
        """Return the last action taken to reach this state, or None if this is the initial state"""

        return self.actions[len(self.actions) - 1] if len(self.actions) > 0 else None

    def heuristic_cost(self) -> float:
        """Return the cost plus the heuristic of this state"""
        return self.cost + self.heuristic

    def solved(self) -> bool:
        """The game is solved if all bolts are solved"""
        return all(bolt.solved() for bolt in self.bolts)

    def __hash__(self):
        """
        The GameState hash is only a function of the bolts.

        Actions and cost are considered metadata and are not included.
        """
        return hash(tuple(slot for bolt in self.bolts for slot in bolt._slots))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return False
        return all([i in other.bolts for i in self.bolts])

    def __gt__(self, other: "GameState") -> bool:
        return self.heuristic_cost() > other.heuristic_cost()

    def __ge__(self, other: "GameState") -> bool:
        return self.heuristic_cost() >= other.heuristic_cost()

    def __lt__(self, other: "GameState") -> bool:
        return self.heuristic_cost() < other.heuristic_cost()

    def __le__(self, other: "GameState") -> bool:
        return self.heuristic_cost() <= other.heuristic_cost()


@dataclass
class Game:
    """The main game class, managing state and game logic"""

    state_name: str
    """The name of the game state for saving/loading"""

    game_state: GameState
    """The current game state"""

    initial_state_str: str = field(init=False, hash=False, compare=False)
    """A string representation of the initial game state, used for visualization"""

    def __post_init__(self):
        self.initial_state_str = str(self)

    def get_stats(self) -> str:
        """
        Game statistics for the current game state,
        including total cost, total steps, and average step cost
        """

        return "\n".join(
            [
                self.get_state_change_str(),
                f"\nTotal cost: {self.game_state.cost:.2f}",
            ]
        )

    def get_state_change_str(self) -> str:
        """
        State change for the current game state,
        showing the initial and current state side by side
        """

        return "\n".join(
            [
                f"{initial}  =>  {current}"
                for initial, current in zip(
                    self.initial_state_str.split("\n"), str(self).split("\n")
                )
            ]
        )

    def snapshot(self) -> GameState:
        """
        Create a snapshot of the current (or next) game state.

        The snapshot is a copy of the game state, including a copy of the bolts, actions,
        and cost, which can be used to restore the game state later.
        """

        return deepcopy(self.game_state)

    def restore_snapshot(self, snapshot: GameState) -> None:
        """
        Restore the game state from a snapshot.
        """

        self.game_state = deepcopy(snapshot)

    def swap_nuts(self, bolt_from_idx: int, bolt_to_idx: int) -> int:
        """
        Swap nuts from one bolt to another.

        Solved bolts will not have nuts removed, and will result in a value of 0 being returned

        Returns the number of bolts moved between 0 and 3. A value of 0 indicates a failed swap.
        """

        # validate inputs
        if (
            bolt_from_idx == bolt_to_idx
            or bolt_from_idx < 0
            or bolt_to_idx >= len(self.game_state.bolts)
        ):
            return 0

        # check what would be popped from the source bolt
        bolt_from = self.game_state.bolts[bolt_from_idx]
        bolt_to = self.game_state.bolts[bolt_to_idx]
        max_nuts = bolt_to.free_slots()

        if bolt_from.solved() or max_nuts == 0:
            return 0
        nuts = bolt_from.nuts()
        if len(nuts) == 0:
            return 0

        # starting from the first nut, pop all matching nuts up to the amount the receiving bolt can accept, replacing with 0s
        count = 1
        first = nuts[0]
        color = bolt_from._slots[first]

        # fail fast for the wrong color
        if not bolt_to.can_accept_nuts((1, color)):
            return 0

        for i in range(first + 1, min(4, first + max_nuts)):
            if bolt_from._slots[i] != color:
                break
            count += 1

        # swap the nuts, collapsing any empty spaces
        bolt_from._slots[first : first + count] = [0] * count
        bolt_to._slots[0:count] = [color] * count
        bolt_to.collapse_spaces()

        self.game_state.actions.append((bolt_from_idx, bolt_to_idx))

        # return the number of nuts swapped
        return count

    def solved(self) -> bool:
        """
        The game is solved if the current game state is solved.
        """

        return self.game_state.solved()

    def visualize(self) -> np.ndarray:
        """
        Return a 2D numpy array representing the game board, with each column
        corresponding to a bolt and each row corresponding to a slot.
        """

        return np.transpose(np.array([bolt._slots for bolt in self.game_state.bolts]))

    def build_action_indicator(self, action: tuple[int, int] | None) -> str:
        if action is not None:
            # convert indicies to account for spacing
            idx_from, idx_to = np.array(action) * 2
            right_side = 2 * len(self.game_state.bolts) - 1

            if idx_from < idx_to:
                # source: left, destination: right
                action_indicator = (
                    (" " * idx_from)
                    + "^"
                    + ("-" * (idx_to - idx_from - 1))
                    + "v"
                    + (" " * (right_side - idx_to - 1))
                )
            elif idx_to < idx_from:
                # source: right, destination: left
                action_indicator = (
                    (" " * idx_to)
                    + "v"
                    + ("-" * (idx_from - idx_to - 1))
                    + "^"
                    + (" " * (right_side - idx_from - 1))
                )
            else:
                # source and destination are the same - should'nt happen
                action_indicator = " " * idx_to + "*" + " " * (right_side - idx_to - 1)

        else:
            # no action
            action_indicator = " " * (2 * len(self.game_state.bolts) - 1)

        return action_indicator

    def display_string(self, formatter: Callable[[int], str] = str) -> str:
        """
        Return a multiline string representation of the game board.

        Includes an indication of the last action taken, if any.
        """

        # format the board as a list of strings
        board: list[list[int]] = self.visualize().tolist()
        rows = [" ".join(formatter(nut) for nut in row) for row in board]

        # build the action indicator
        action_indicator = self.build_action_indicator(self.game_state.last_action())

        return "\n".join(
            [action_indicator, *rows, "-" * (2 * len(self.game_state.bolts) - 1)]
        )

    def colored_string(self) -> str:
        """
        Return `display_string` formatted with colors.

        Only suitable for terminal outputs supporting ANSI color codes.
        """

        return self.display_string(lambda nut: colored_nut_str(nut))

    def __str__(self) -> str:
        """
        Return `display_string` with no additional formatting.

        Suitable for terminal outputs without color support, or writing to a text file.
        """

        return self.display_string()

    @staticmethod
    def random_state(bolt_count: int) -> "Game":
        """
        Return a randomly generated game state.
        """

        min_empty = 2

        bolts = np.array(
            [np.full(4, (i % 9) + 1) for i in range(bolt_count - min_empty)]
        )
        bolts = permutation(bolts.ravel()).reshape(bolts.shape)

        return Game.from_state(f"random{bolt_count}", bolts.tolist(), bolt_count)

    @staticmethod
    def from_state(name: str, state: list[list[int]], n_bolts: int) -> "Game":
        bolts = [Bolt(slots) for slots in state]

        # ensure the number of bolts does not exceed n_bolts
        if len(bolts) > n_bolts:
            raise ValueError(
                f"Invalid state file: {name}. Too many bolts! (should be at most {n_bolts}, was {len(bolts)})"
            )

        # create a game state, adding empty bolts to have n_bolts total
        game = Game(
            name,
            GameState(bolts + [Bolt([0, 0, 0, 0]) for _ in range(len(bolts), n_bolts)]),
        )

        # validate that the game is solvable
        all_nuts = [slot for bolt in game.game_state.bolts for slot in bolt._slots]
        for color in set(all_nuts):
            if all_nuts.count(color) % 4 != 0:
                raise ValueError(
                    f"Invalid state file: {name}. Color {color} has an invalid count {all_nuts.count(color)} (expected multiple of 4)"
                )

        # validate each bolt individually
        valid_bolts = [bolt.validate(initial=True) for bolt in game.game_state.bolts]
        if not all(valid_bolts):
            raise ValueError(
                f"Invalid state file: {name}. Invalid bolts {[i for i, valid in enumerate(valid_bolts) if not valid]}"
            )

        return game

    @staticmethod
    def from_state_file(name: str) -> "Game":
        """
        Load a game state from a file located in the `states` directory.

        The filename should not contain an extension
        """

        with open(f"states/{name}.txt") as file:
            # get all the lines from the file, stripping whitespace and ignoring comments
            lines = [
                line.strip()
                for line in file.readlines()
                if not line.startswith("#") and line.strip() != ""
            ]

            # need at least 2 lines: n_bolts and at least one bolt
            if len(lines) < 2:
                raise ValueError(f"Invalid state file: {name}")

            # get the number of bolts from the first line
            n_bolts = int(lines[0].strip())

            # create bolts from the remaining lines

            return Game.from_state(
                name,
                [[int(nut) for nut in slots.split()] for slots in lines[1:]],
                n_bolts,
            )

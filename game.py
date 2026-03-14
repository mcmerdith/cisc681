from copy import deepcopy
from dataclasses import dataclass, field
from functools import reduce
from typing import Callable

import numpy as np
from termcolor import colored, cprint

COLORS = ["default", "yellow", "red", "blue", "green", "magenta", "cyan"]


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

    _previous_slots: list[int] = field(init=False, hash=False, compare=False)
    """The previous state of the slots, used for undoing moves"""

    def __post_init__(self):
        self._previous_slots = self._slots.copy()

    def nuts(self) -> list[int]:
        """Get the indices of nuts on the bolt"""

        return [i for i, slot in enumerate(self._slots) if slot != 0]

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

    def push(self, nuts: tuple[int, int]) -> bool:
        """
        Push nuts onto the bolt.

        nuts is a tuple of (count, color).

        Returns True if the push was successful, False otherwise.
        """

        count, color = nuts

        # pushing nothing is a no-op
        if count <= 0 or color <= 0:
            return True

        if not self.can_accept_nuts(nuts):
            return False

        # save state
        self._previous_slots = self._slots.copy()

        # push the nuts onto the bolt, collapsing any empty spaces
        self._slots[0:count] = [color for i in range(count)]
        self.collapse_spaces()

        return True

    def pop(self) -> tuple[int, int]:
        """
        Pop all nuts of one color from the top of the bolt.

        Returns the nuts that were popped, or an empty list if the bolt is empty.
        """

        # save state
        self._previous_slots = self._slots.copy()

        # get the nuts to pop
        nuts = self.nuts()
        if len(nuts) == 0:
            return (0, 0)

        # starting from the first nut, pop all matching nuts, replacing with 0s
        count = 0
        color = self._slots[nuts[0]]
        for i in range(nuts[0], 4):
            if count != 0 and self._slots[i] != color:
                break
            count += 1
            self._slots[i] = 0

        # return the resulting popped nuts
        return (count, color)

    def undo(self):
        """
        Undo the last operation by restoring the previous state.

        Calling more than once will have no additional effect.
        """

        self._slots = self._previous_slots.copy()

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
            if slot == 0 and not all(s == 0 for s in acc):
                # empty spaces may only be preceeded by empty spaces
                return acc + [-1]
            return acc + [slot]

        valid_slots = reduce(check_nuts, self._slots, [])
        if any(slot < 0 for slot in valid_slots):
            cprint(
                f"Invalid state: bolt has an illegal space at {[i for i, slot in enumerate(valid_slots) if slot < 0]}",
                "red",
            )
            return False

        return True


@dataclass()
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

    def last_action(self) -> tuple[int, int] | None:
        """Return the last action taken to reach this state, or None if this is the initial state"""

        return self.actions[len(self.actions) - 1] if len(self.actions) > 0 else None

    def solved(self) -> bool:
        """The game is solved if all bolts are solved"""
        return all(bolt.solved() for bolt in self.bolts)

    def __hash__(self):
        """
        The GameState hash is only a function of the bolts.

        Actions and cost are considered metadata and are not included.
        """
        return hash(tuple(slot for bolt in self.bolts for slot in bolt._slots))

    def __gt__(self, other: "GameState") -> bool:
        return self.cost > other.cost

    def __ge__(self, other: "GameState") -> bool:
        return self.cost >= other.cost

    def __lt__(self, other: "GameState") -> bool:
        return self.cost < other.cost

    def __le__(self, other: "GameState") -> bool:
        return self.cost <= other.cost


# tests to ensure GameState equality, inequality, and hash work correctly
# solvers rely on these to check if a state has already been checked
test_state_A = GameState([Bolt([0, 0, 0, 0])], [], 0.0)
test_state_B = GameState([Bolt([0, 0, 0, 0])], [(1, 4)], 3.0)
test_state_C = GameState([Bolt([0, 1, 1, 3])], [], 0.0)
assert test_state_A == test_state_B, "GameState equality failed"
assert test_state_A != test_state_C, "GameState inequality failed"
assert test_state_A.solved(), "GameState solve check failed"
assert not test_state_C.solved(), "GameState solve check failed"
assert hash(test_state_A) == hash(test_state_B), "GameState hash check failed"
assert hash(test_state_A) != hash(test_state_C), "GameState hash check failed"


@dataclass
class Game:
    """The main game class, managing state and game logic"""

    state_name: str
    """The name of the game state for saving/loading"""

    game_state: GameState
    """The current game state"""

    next_game_state: GameState = field(
        init=False, repr=False, hash=False, compare=False
    )
    """The result of a dry-run move, or the current state if no move has been dry-run"""

    initial_state_str: str = field(init=False, hash=False, compare=False)
    """A string representation of the initial game state, used for visualization"""

    def __post_init__(self):
        self.initial_state_str = str(self)
        self.next_game_state = deepcopy(self.game_state)

    def get_stats(self) -> str:
        """
        Game statistics for the current game state,
        including total cost, total steps, and average step cost
        """

        return "\n".join(
            [
                f"Total cost: {self.game_state.cost:.2f}",
                f"Total steps: {len(self.game_state.actions)}",
                f"Average step cost: {self.game_state.cost / len(self.game_state.actions):.2f}",
            ]
        )

    def snapshot(self, next_state=False) -> GameState:
        """
        Create a snapshot of the current (or next) game state.

        The snapshot is a copy of the game state, including a copy of the bolts, actions,
        and cost, which can be used to restore the game state later.
        """

        return deepcopy(self.next_game_state if next_state else self.game_state)

    def restore_snapshot(self, snapshot: GameState) -> None:
        """
        Restore the game state from a snapshot.
        """

        self.game_state = deepcopy(snapshot)
        self.next_game_state = deepcopy(snapshot)

    def swap_nut(self, bolt_from: int, bolt_to: int, dry_run=False) -> int:
        """
        Swap a nut from one bolt to another.

        If `dry_run` is True, the swap is performed on a copy of the bolts
        and stored to `next_game_state`, leaving the current game state unchanged.

        Returns the number of bolts moved. A value of 0 indicates a failed swap.
        """

        # validate inputs
        if (
            bolt_from == bolt_to
            or bolt_from < 0
            or bolt_to >= len(self.game_state.bolts)
        ):
            return 0

        # select either the current state or create a dry-run copy
        if dry_run:
            self.next_game_state = deepcopy(self.game_state)
            state = self.next_game_state
        else:
            state = self.game_state

        # attempt to pop nuts off the source bolt
        nuts = state.bolts[bolt_from].pop()
        count, _ = nuts
        if count == 0:
            return 0

        # attempt to push the popped nuts onto the target bolt
        if not (state.bolts[bolt_to].push(nuts)):
            # undo the pop if push fails
            state.bolts[bolt_from].undo()
            return 0

        # save the action to the game state
        state.actions.append((bolt_from, bolt_to))

        # reset the dry-run state if a successful non-dry-run swap occurs
        if not dry_run:
            self.next_game_state = self.game_state

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

    def display_string(self, formatter: Callable[[int], str] = str) -> str:
        """
        Return a multiline string representation of the game board.

        Includes an indication of the last action taken, if any.
        """

        # format the board as a list of strings
        board: list[list[int]] = self.visualize().tolist()
        rows = [" ".join(formatter(nut) for nut in row) for row in board]

        # build the action indicator, if any
        action = self.game_state.last_action()
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
    def from_state(name: str) -> "Game":
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
            bolts = [Bolt([int(nut) for nut in line.split()]) for line in lines[1:]]

            # ensure the number of bolts does not exceed n_bolts
            if len(bolts) > n_bolts:
                raise ValueError(
                    f"Invalid state file: {name}. Too many bolts! (should be at most {n_bolts}, was {len(bolts)})"
                )

            # create a game state, adding empty bolts to have n_bolts total
            game = Game(
                name,
                GameState(
                    bolts + [Bolt([0, 0, 0, 0]) for _ in range(len(bolts), n_bolts)]
                ),
            )

            # validate each bolt individually
            valid_bolts = [
                bolt.validate(initial=True) for bolt in game.game_state.bolts
            ]
            if not all(valid_bolts):
                raise ValueError(
                    f"Invalid state file: {name}. Invalid bolts {[i for i, valid in enumerate(valid_bolts) if not valid]}"
                )

            return game

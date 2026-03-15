from termcolor.termcolor import cprint

from game import Bolt, Game, GameState

# tests to ensure GameState equality, inequality, and hash work correctly
# solvers rely on these to check if a state has already been checked
cprint("Testing that all GameState comparisons work correctly", "light_blue")
test_state_A = GameState([Bolt([0, 0, 0, 0])], [], 0.0)
test_state_B = GameState([Bolt([0, 0, 0, 0])], [(1, 4)], 3.0)
test_state_C = GameState([Bolt([0, 1, 1, 3])], [], 0.0)
assert test_state_A == test_state_B, "GameState equality failed"
assert test_state_A != test_state_C, "GameState inequality failed"
assert test_state_A < test_state_B, "GameState comparison failed"
assert test_state_A <= test_state_C, "GameState comparison failed"
assert test_state_B > test_state_A, "GameState comparison failed"
assert test_state_C >= test_state_A, "GameState comparison failed"
assert test_state_A.solved(), "GameState solve check failed"
assert not test_state_C.solved(), "GameState solve check failed"
assert hash(test_state_A) == hash(test_state_B), "GameState hash check failed"
assert hash(test_state_A) != hash(test_state_C), "GameState hash check failed"

# tests to ensure Game obeys the rules
cprint("Testing that all Game swap operations work correctly", "light_blue")
initial_state = GameState(
    [
        Bolt([2, 1, 1, 2]),
        Bolt([2, 2, 1, 1]),
        Bolt([3, 3, 3, 3]),
        Bolt([0, 0, 0, 0]),
        Bolt([0, 0, 0, 0]),
    ]
)

game = Game("testing", initial_state)
cprint(game.colored_string())
assert game.swap_nuts(3, 4) == 0, "swap_nuts failed"
assert initial_state == game.game_state, "swap_nuts failed"
assert game.swap_nuts(3, 1) == 0, "swap_nuts failed"
assert initial_state == game.game_state, "swap_nuts failed"

move_one_state = GameState(
    [
        Bolt([0, 1, 1, 2]),
        Bolt([2, 2, 1, 1]),
        Bolt([3, 3, 3, 3]),
        Bolt([0, 0, 0, 2]),
        Bolt([0, 0, 0, 0]),
    ]
)
move_one = game.swap_nuts(0, 3)
cprint(game.colored_string())
assert move_one == 1, "swap_nuts failed"
assert move_one_state == game.game_state, "swap_nuts failed"

move_two_state = GameState(
    [
        Bolt([0, 1, 1, 2]),
        Bolt([0, 0, 1, 1]),
        Bolt([3, 3, 3, 3]),
        Bolt([0, 2, 2, 2]),
        Bolt([0, 0, 0, 0]),
    ]
)
move_two = game.swap_nuts(1, 3)
cprint(game.colored_string())
assert move_two == 2, "swap_nuts failed"
assert move_two_state == game.game_state, "swap_nuts failed"

move_three_state = GameState(
    [
        Bolt([1, 1, 1, 2]),
        Bolt([0, 0, 0, 1]),
        Bolt([3, 3, 3, 3]),
        Bolt([0, 2, 2, 2]),
        Bolt([0, 0, 0, 0]),
    ]
)
move_three = game.swap_nuts(1, 0)
cprint(game.colored_string())
assert move_three == 1, "swap_nuts failed"
assert move_three_state == game.game_state, "swap_nuts failed"

move_four_state = GameState(
    [
        Bolt([0, 0, 0, 2]),
        Bolt([1, 1, 1, 1]),
        Bolt([3, 3, 3, 3]),
        Bolt([0, 2, 2, 2]),
        Bolt([0, 0, 0, 0]),
    ]
)
move_four = game.swap_nuts(0, 1)
cprint(game.colored_string())
assert move_four == 3, "swap_nuts failed"
assert move_four_state == game.game_state, "swap_nuts failed"

move_five_state = GameState(
    [
        Bolt([0, 0, 0, 0]),
        Bolt([1, 1, 1, 1]),
        Bolt([3, 3, 3, 3]),
        Bolt([2, 2, 2, 2]),
        Bolt([0, 0, 0, 0]),
    ]
)
move_five = game.swap_nuts(0, 3)
cprint(game.colored_string())
assert move_five == 1, "swap_nuts failed"
assert move_five_state == game.game_state, "swap_nuts failed"

assert game.swap_nuts(1, 0) == 0, "swap_nuts failed"
assert move_five_state == game.game_state, "swap_nuts failed"
assert game.swap_nuts(1, 2) == 0, "swap_nuts failed"
assert move_five_state == game.game_state, "swap_nuts failed"

assert game.solved(), "solved() failed"

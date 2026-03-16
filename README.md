# CISC681 Program 1

[https://github.com/mcmerdith/cisc681/tree/hw1](https://github.com/mcmerdith/cisc681/tree/hw1)

### Matthew Meredith

## Notes

There is a lot more here than is necessary for assignment completion.
Most of the code for the assignment is in `uninformed.py` and `informed.py`.
There are a few things in `game.py` that are relevant (namely `GameState`,
the features of which are used extensively by the search algorithms)

I enjoyed this assignment, and found it was a good opportunity to reinforce
my understanding of the search algorithms, while also practicing building
software in Python, so there's a lot of extra stuff.

### Core Features

  - Breadth-First Search
  - A\*, Iterative Deepening A\*

### Extra Features

They're arguably not needed, but they made it a lot easier for me to catch bugs and
figure out what was going wrong when a solver wasn't doing what I expected.
(and they're just cool - much more interesting to watch a solver work than just see an output)

  - Advanced visualization
    - print intermediate states to the console (replacing the previous printout)
    - colorized view of the game state
  - Replay: view a previous solution with only the states that were used to achieve the goal
  - Task runner: run multiple solvers on multiple problems with one command
  - Random state generation: push the algorithms to the limits without having to manually create states
  - A Solver API to share some common functions between any type of iterative solver
  - A rigid Game API to handle game state and rules
    - less work required for the solver to optimize the search space
    - prevents me from being a bonehead (usually)

## Quick Start

Install Dependencies: `pip install numpy termcolor colorama`

> **Note**:
>
> `run.py` is the primary runner script capable of solving states or replaying solutions.
> See `python run.py --help` for usage.
>
> `uninformed.py` and `informed.py` will run only the assignment specified requirements.

```console
# Uninformed Search (BFS)
python uninformed.py problemN

# Informed Search (IDA*)
python informed.py problemN
```

Pre-computed solutions are available in `solutions/`. Running a solver again will overwrite its existing solution.
The computed solution should not change, but the runtime in `stats/` may be marginally different.

You can visualize solutions by running `python run.py <...solvers> --replay --states <...problems>`.
Omitting `--states` will replay all available solutions for the specified solvers.

## A. Search Space

Given `n+2` bolts, there are `4*(n+2)` possible slots for each nut to be placed.

I would calculate the total search space as `Permutation(4*(n+2), 4*n)`. This is an overestimate, as it considers all possible arrangements even though many are invalid (spaces between nuts on a bolt). In my implementation, I have added an optimization to not consider the order of the bolts, which further reduces the search space. However, I am bad at math so I have not calculated the reduction in search space of this optimization.

## B. Max Branching Factor

Given `n+2` bolts, there are `(n+2)^2` possible selection for which pair of bolts to swap. Since it doesn't make sense to swap a bolt with itself, we reduce this by `n+2`, giving us `(n+2)^2 - (n+2)`. This assumes there is an available slot on each bolt, which is often not the case so the average branching factor is much lower in most cases.

## C. Uninformed Search

I implemented a *Breadth First Search* algorithm. I chose it because we are trying to find
the shortest path, and BFS is guaranteed to find it, since all our step costs are 1.

I was able to push the game size up to 8 bolts (2 empty bolts) before the time taken started to significantly increase.

#### Problem 1

```
States expanded: 614
Max fringe size: 718
Time taken     : 0.28s
```

#### Problem 2

```
States expanded: 290
Max fringe size: 391
Time taken     : 0.13s
```

#### Problem 3

```
States expanded: 14330
Max fringe size: 15711
Time taken     : 10.69s
```

### D. Heuristic

The heuristic calculates the minimum number of moves that are required to move all nuts of a given color onto a minimum number of solved bolts.

It is calculated as the number of different bolts that contain them minus the number of solved bolts that could contain them (one nut is assumed to already be on the solution bolt)

This is admissible because it assumes that every move constructs the solution, when some moves may be impossible (2 nuts on the same bolt separated by other nuts are counted as one move when that is impossible, if there are no nuts on the bottom of a bolt then a minimum of 1 move per solved bolt is required because a solution must start from the bottom, etc)

I was able to push A* to a game size of 12 bolts (2 empty bolts) before the time taken started to significantly increase.

I was only able to push my IDA* implementation to a game size of 7 bolts (2 empty bolts) before the time taken started to significantly increase. I would assume that when the goal depth is high many nodes that would have already been pruned by A* are re-expanded by IDA*.

If the code was more efficient I probably could have gone further. However, the actual algorithm code is massively overshadowed by the copying overhead of saving and restoring game states, so I would have to completely redesign my implementation, and I've rewritten it several times already to get better performance (probably already more than necessary for this assignment).

## E. Informed Search

I implemented both the **Iterative Deepening A\*** and **A\*** algorithms.

#### Problem 1

```
IDA*
States expanded: 76
Recursion depth: 8
Time taken     : 0.03s

A*
States expanded: 85
Max fringe size: 253
Time taken     : 0.05s
```

#### Problem 2

```
IDA*
States expanded: 14
Recursion depth: 6
Time taken     : 0.0s

A*
States expanded: 36
Max fringe size: 102
Time taken     : 0.02s
```

#### Problem 3

```
IDA*
States expanded: 547
Recursion depth: 12
Time taken     : 0.32s

A*
States expanded: 167
Max fringe size: 478
Time taken     : 0.16s
```

## AI Statement

A local model was used to generate inline completions.

All code (game code and search algorithms) is primarily human work.

## Problem Overview

The goal of the puzzle is to have all nuts of a matching color on the same bolt.
A bolt can only hold 4 nuts. Only the top nut(s) can be moved from a bolt, and if they
are matching colors, all nuts move in one action. Nuts can only be moved to a bolt that
is empty, or has a matching color on top, and enough room (only 4 nuts per bolt).

Your programs should take as an argument the name of a file that has a specification of
the start state (detailed below) and produce (to stdout) a solution (series of moves from
start to a goal state) in the format discussed below.

## Input file and output formatting

### Input file format
```
# this is a comment
<number of bolts>
<colors of bolt 1 (top to bottom)>
<colors of bolt 2 top to bottom>
...
```

### Solution format
```
<from bolt> <to bolt>
<from bolt> <to bolt>
...
```

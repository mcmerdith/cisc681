# CISC681 Program 1

[https://github.com/mcmerdith/cisc681/tree/bolt-puzzle](https://github.com/mcmerdith/cisc681/tree/bolt-puzzle)

### Matthew Meredith

For the original assignment repo, see:

[https://github.com/mcmerdith/cisc681/tree/program-1](https://github.com/mcmerdith/cisc681/tree/program-1)

## Features

  - Breadth-First Search, A\*, Iterative Deepening A\*
  - Interactive Mode
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

Install Dependencies: `pip install numpy termcolor colorama blessed`

> **Note**:
>
> `run.py` is the primary runner script capable of solving states or replaying solutions.
> See `python run.py --help` for usage.

```console
# Example commands
> python run.py interactive --no-save --random 6
> python run.py astar idastar --states problem3
> python run.py bfs --replay --states problem1 problem2

# Full usage
> python run.py --help
usage: run.py [-h] [--clean] [--states STATES [STATES ...]] [--random RANDOM] [--replay] [--no-save] [--max-iterations MAX_ITERATIONS] [--print-steps] [--step-delay-ms STEP_DELAY_MS]
              {gamblers,bfs,idastar,astar,interactive} [{gamblers,bfs,idastar,astar,interactive} ...]

positional arguments:
  {gamblers,bfs,idastar,astar,interactive}
                        List of solvers to use

options:
  -h, --help            show this help message and exit
  --clean               Remove all old solution files for SOLVERS before solving
  --states STATES [STATES ...]
                        List of names (without extension) of problem states to solve (located in states/)
  --random RANDOM       Solve a random state with n_bolts=RANDOM. Overrides --states
  --replay              Replay solutions from STATES and SOLVERS. Cannot be used with --random
  --no-save             Do not save the solution. Solutions are saved by default
  --max-iterations MAX_ITERATIONS
                        Maximum number of iterations to run. Solvers not complete within MAX_ITERATIONS will be considered failed
  --print-steps         Print the state after each step. Recommended to use with --step-delay-ms for visualization
  --step-delay-ms STEP_DELAY_MS
                        Delay between steps in milliseconds
```

Pre-computed solutions are available in `solutions/`. Running a solver again will overwrite its existing solution.
The computed solution should not change, but the runtime in `stats/` may be marginally different.

You can visualize solutions by running `python run.py <...solvers> --replay --states <...problems>`.
Omitting `--states` will replay all available solutions for the specified solvers.

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

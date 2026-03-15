# CISC681 Repo - HW1

### Matthew Meredith

## Quick Start

Install Dependencies: `pip install numpy termcolor colorama`

Run all solvers on all problems: `python run.py`

Run uninformed search only: `python run.py --solvers bfs` or `python uninformed.py <problem name>`

Run informed search only: `python run.py --solvers idastar astar` or `python informed.py <problem name>`

### Runner Parameters

```
usage: run.py [-h] [--clean] [--states STATES [STATES ...]] [--solvers SOLVERS [SOLVERS ...]] [--random RANDOM] [--print-state] [--no-save]

options:
  -h, --help            show this help message and exit
  --clean               Remove all old solution files before solving
  --states STATES [STATES ...]
                        List of problem states to solve
  --solvers SOLVERS [SOLVERS ...]
                        List of solvers to use
  --random RANDOM       Solve a random state with RANDOM bolts
  --print-state         Print the state after each step
  --no-save             Do not save the solution
```

## Problem Overview

The goal of the puzzle is to have all nuts of a matching color on the same bolt.
A bolt can only hold 4 nuts. Only the top nut(s) can be moved from a bolt, and if they
are matching colors, all nuts move in one action. Nuts can only be moved to a bolt that
is empty, or has a matching color on top, and enough room (only 4 nuts per bolt).

Your programs should take as an argument the name of a file that has a specification of
the start state (detailed below) and produce (to stdout) a solution (series of moves from
start to a goal state) in the format discussed below.

## Uninformed Search

I implemented a *Breadth First Search* algorithm. I chose it because we are trying to find
the shortest path, and BFS is guaranteed to find it, since all our step costs are 1.

## IDA*
TODO

## AI Statement

A local-LLM was used to generate some inline completions.

All code (game, search algorithms) is primarily human work.

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

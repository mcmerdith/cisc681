# CISC681 Repo - HW1

### Matthew Meredith

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

# CISC681 Program 2

### Matthew Meredith

## Quick start

```bash
# Install dependencies
pip install swig gymnasium "gymnasium[toy-text]" tqdm`
```

### Structure

`qlearn.py`: The main agent algorithm

`test.py`: A wrapper to train and test agents

`utils.py`: Performance measurement, parameter wrappers, math functions, and string utilities

## Report

### A. Implement Q-Learning

See [qlearn.py](qlearn.py)

```bash
# Run with parameters set from the command line
python qlearn.py --help
# Sample run: 8x8 with success rate of 0.75
python qlearn.py --test-agent --convergence-method policy_convergence --world-size 8 --success-rate 0.75
```

### B. Exploration vs Exploitation

I implemented an "optimistic utility" function that encourages visiting positions
which have been visited less (Lecture 11, Slide 6).

Additionally, I exponentially decay the probability of taking a random action,
so that less exploration occurs as training nears completion

### C. Deterministic statistics

```bash
# Prints stats for combinations of 4x4, 8x8, deterministic, stochastic, fixed and random maps
python test.py
```

```
4x4:
  ~750 iterations
  stddev = ~30

8x8
  ~5500 iterations
  stddev = ~100
```

### D. Stochastic statistics

```bash
# Prints stats for combinations of 4x4, 8x8, deterministic, stochastic, fixed and random maps
python test.py
```

```
4x4:
  ~2500 iterations
  stddev = ~1100

8x8
  ~10500 iterations
  stddev = ~1500
```

### E. Determining if an environment is learned

For both deterministic and stochastic environments, I used policy convergence over a
window of 500 iterations.

For stochastic environments, there is too much noise to use q-convergence.

Q-convergence does work for deterministic environments, but provides very similar results
to policy convergence.

### F.iii Changing reward structure

With the exception of a deterministic 4x4, agents struggled or failed to learn with
small reward values (1, -1). All success rates were less than 20%, most failed entirely.

With larger rewards (10, -10), agents performed much better, but still struggled or
failed to learn without a small negative living reward (-0.001). Agents without a negative
living reward all failed with the exception of the 4x4 stochastic, which succeeded ~40% of
the time.

### F.ii Changing success rate

> I already wrote the code to do these parameter combinations, so I just did this one too

Numbers are relative to a change in success rate from `0.5 -> 0.95`

Success rate generally improved with a higher success rate:
`4x4: 40% -> 85%, 8x8: 40% -> 92%`

The agents also took less iterations on average to converge:
`4x4: 2100 -> 1400, 8x8: 17000 -> 8000`

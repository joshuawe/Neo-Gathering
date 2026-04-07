# Neo-Gathering
A redesigned "Resource Gathering" environment for Reinforcement Learning in the gymnasium framework.

We take the original Resource Gathering environment and improve it:
- [x] We can randomly generate maps, by setting a seed.
- [x] The map size can be defined.
- [x] The number of dragons, gold and silver are .
- [x] The observation space is a fixed area around the agent (5x5 by default)
- [ ] Computationally faster -> env vectorization
- [ ] Computationally faster -> env vectorization (JAX implementation planned.)
- [x] benchmark the difference in speed using `gymnasium.utils.performance.benchmark_step` and co
- [ ] Scalar objective `neo-gathering` and multiple objective env `mo-neo-gathering` (this is basically the same env with one flag set differenctly)


# Installation

Install this repository with `uv`

```
uv add "git+https://github.com/joshuawe/Neo-Gathering"
```

(or using `pip`: `pip install "git+https://github.com/joshuawe/Neo-Gathering"`)



# Development & Contribution

Tests are written in the `pytest` framework and stored under `tests/`. Use `uv run pytest` to run all tests.


# Benchmark

Measured with `gymnasium.utils.performance` (`target_duration=10s`, `seed=42`) on a 5×5 map with a 5×5 observation window. Run `uv run python scripts/benchmark.py` to reproduce.

| Metric | neo-gathering | resource-gathering | ratio |
|---|---|---|---|
| Init (inits/s) | 2 590 | 3 691 | 0.70x |
| Step (steps/s) | 117 163 | 174 720 | 0.67x |
| Render (frames/s) | 1 597 | 1 729 | 0.92x |

**Notes**

- The step gap is structural: resource-gathering returns a flat 4-integer observation `(x, y, has_gold, has_gem)`, while neo-gathering returns a full 5×5 window slice — roughly 6× more data per step.
- Init is slower primarily due to random map generation (`create_map`) and pre-computing the padded map on each reset.
- Render performance is nearly on par (0.92x).
- Further throughput gains are best achieved via vectorised environments (`gymnasium.vector`) or a future JAX implementation, rather than single-env Python optimisation.
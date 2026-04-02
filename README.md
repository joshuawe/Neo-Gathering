# Neo-Gathering
A redesigned "Resource Gathering" environment for Reinforcement Learning.

We take the original Resource Gathering environment and improve it:
- [ ] We can randomly generate maps, by setting a seed.
- [ ] The map size can be defined.
- [ ] The number of dragons, gold and silver are .
- [ ] The observation space is a fixed area around the agent (5x5 by default)
- [ ] Computationally faster -> env vectorization
- [ ] Computationally faster -> JAX implementation
- [ ] Computationally faster -> env vectorization (JAX implementation planned.)
- [ ] benchmar the difference in speed using `gymnasium.utils.performance.benchmark_step` and co

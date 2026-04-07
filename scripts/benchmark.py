"""
Benchmark neo-gathering-v0 vs resource-gathering-v0 using gymnasium performance utilities.
"""
import gymnasium as gym
# import mo_gymnasium  # noqa: F401 — registers resource-gathering-v0
import neo_gathering  # noqa: F401 — registers neo-gathering-v0

from gymnasium.utils.performance import benchmark_init, benchmark_step, benchmark_render

SEED = 42
DURATION = 10  # seconds per benchmark

# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_neo():
    env = gym.make("neo-gathering-v0", render_mode=None, obs_window=(5, 5))
    _ = env.reset()
    return env

def make_rg():
    env = gym.make("resource-gathering-v0", render_mode=None)
    _ = env.reset()
    return env

def make_neo_rgb():
    env = gym.make("neo-gathering-v0", render_mode="rgb_array", obs_window=(5, 5))
    _ = env.reset()
    return env

def make_rg_rgb():
    env = gym.make("resource-gathering-v0", render_mode="rgb_array")
    _ = env.reset()
    return env

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

print("=" * 60)
print("gymnasium performance benchmark")
print(f"  target_duration = {DURATION}s per test, seed = {SEED}")
print("=" * 60)

# --- benchmark_init ---------------------------------------------------------
print("\n[benchmark_init]  (env initializations + first reset per second)")

neo_init = benchmark_init(make_neo, target_duration=DURATION, seed=SEED)
rg_init  = benchmark_init(make_rg,  target_duration=DURATION, seed=SEED)

print(f"  neo-gathering : {neo_init:.1f} inits/s")
print(f"  resource-gathering : {rg_init:.1f} inits/s")
print(f"  ratio (neo / rg) : {neo_init / rg_init:.2f}x")

# --- benchmark_step ---------------------------------------------------------
print("\n[benchmark_step]  (steps per second)")

env_neo = make_neo()
env_rg  = make_rg()

neo_step = benchmark_step(env_neo, target_duration=DURATION, seed=SEED)
rg_step  = benchmark_step(env_rg,  target_duration=DURATION, seed=SEED)

env_neo.close()
env_rg.close()

print(f"  neo-gathering : {neo_step:.1f} steps/s")
print(f"  resource-gathering : {rg_step:.1f} steps/s")
print(f"  ratio (neo / rg) : {neo_step / rg_step:.2f}x")

# --- benchmark_render -------------------------------------------------------
print("\n[benchmark_render]  (render frames per second, rgb_array mode)")

env_neo_rgb = make_neo_rgb()
env_rg_rgb  = make_rg_rgb()

neo_render = benchmark_render(env_neo_rgb, target_duration=DURATION)
rg_render  = benchmark_render(env_rg_rgb,  target_duration=DURATION)

env_neo_rgb.close()
env_rg_rgb.close()

print(f"  neo-gathering : {neo_render:.1f} frames/s")
print(f"  resource-gathering : {rg_render:.1f} frames/s")
print(f"  ratio (neo / rg) : {neo_render / rg_render:.2f}x")

print("\n" + "=" * 60)
print("Done.")

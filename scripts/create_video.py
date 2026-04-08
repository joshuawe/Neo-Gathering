"""
Creates two videos of a random agent interacting with the NeoGathering environment.

Video 1: 5x5 map, 3x3 obs window, 100 steps (home field entry blocked).
Video 2: 10x10 map — obs window animation (3x3 → 5x5 → 7x7 → back to 5x5),
         then 100 steps with home field entry blocked.

Requires dev dependencies: uv sync --group dev
"""

import os

import gymnasium as gym
import imageio
import numpy as np

import neo_gathering  # noqa: F401

OUTPUT_DIR = "figures"
FPS = 12
PAUSE_FRAMES = 3  # frames held per obs-window size during animation (~3 s at 4 fps)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def sample_no_home(env_raw):
    """Return a random action that does not step onto the home cell.

    Falls back to any action if all moves lead to home (e.g. agent is surrounded).
    """
    home = env_raw._get_home_position()
    dirs = env_raw.direction_dict
    rows, cols = env_raw.map_size

    candidates = list(range(env_raw.action_space.n))
    np.random.shuffle(candidates)

    for action in candidates:
        dr, dc = dirs[action]
        nr, nc = env_raw.current_pos[0] + dr, env_raw.current_pos[1] + dc
        in_bounds = 0 <= nr < rows and 0 <= nc < cols
        if in_bounds and (nr, nc) == home:
            continue
        return action
    return env_raw.action_space.sample()  # fallback: all moves blocked


def set_obs_window(env_raw, size: int):
    """Update obs_window and derived attributes for rendering."""
    env_raw.obs_window = (size, size)
    env_raw._ws_x = size
    env_raw._ws_y = size
    env_raw._dx = size // 2
    env_raw._dy = size // 2


def record_steps(env, env_raw, n_steps: int) -> list:
    """Run n_steps with home-blocking policy, resetting on early termination."""
    frames = []
    for _ in range(n_steps):
        action = sample_no_home(env_raw)
        _, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            env.reset()
    return frames


# ---------------------------------------------------------------------------
# Video 1 — 5×5 map, 3×3 obs window, 100 steps
# ---------------------------------------------------------------------------
print("Recording Video 1: 5×5 map, 3×3 obs window, 100 steps …")

env = gym.make(
    "neo-gathering-v0",
    render_mode="rgb_array",
    map_size=(5, 5),
    obs_window=(3, 3),
)
env.reset(seed=42)
env_raw = env.unwrapped

frames = [env.render()]
frames += record_steps(env, env_raw, 50)
env.close()

out_path = os.path.join(OUTPUT_DIR, "neo_gathering_5x5.mp4")
imageio.mimwrite(out_path, frames, fps=FPS)
print(f"  → {out_path}  ({len(frames)} frames)")

# ---------------------------------------------------------------------------
# Video 2 — 10×10 map, obs-window animation then 100 steps
# ---------------------------------------------------------------------------
print("Recording Video 2: 10×10 map, obs-window animation + 100 steps …")

env = gym.make(
    "neo-gathering-v0",
    render_mode="rgb_array",
    map_size=(10, 10),
    obs_window=(3, 3),
    num_gold=10,
    num_silver=5,
    num_dragons=4
)
env.reset(seed=8)
env_raw = env.unwrapped

frames = []

# Pause on initial state (3×3)
frames += [env.render()] * PAUSE_FRAMES

# Grow obs window
for size in (3, 7, 9, 5):
    set_obs_window(env_raw, size)
    frames += [env.render()] * PAUSE_FRAMES
    frames += record_steps(env, env_raw, 10)
    frames += [env.render()] * PAUSE_FRAMES

# Return to 5×5
set_obs_window(env_raw, 5)
frames += [env.render()] * PAUSE_FRAMES

# Agent moves for 100 steps
# frames += record_steps(env, env_raw, 50)
env.close()

out_path = os.path.join(OUTPUT_DIR, "neo_gathering_10x10.mp4")
imageio.mimwrite(out_path, frames, fps=FPS)
print(f"  → {out_path}  ({len(frames)} frames)")

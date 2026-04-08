"""
Creates a GIF of a random agent interacting with the NeoGathering environment,
stitching two clips together:

  Clip 1: 5×5 map, 3×3 obs window, 100 steps (home field entry blocked).
  Clip 2: 10×10 map — obs window animation (3×3 → 5×5 → 7×7 → back to 5×5),
           then 100 steps with home field entry blocked.

Requires dev dependencies: uv sync --group dev
"""

import os

import gymnasium as gym
import imageio
import numpy as np
from PIL import Image

import neo_gathering  # noqa: F401

OUTPUT_DIR = "figures"
FPS = 12
PAUSE_FRAMES = 5   # frames held per obs-window size during animation
GAP_FRAMES = 2     # black frames between the two clips

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


def resize_frames(frames: list, size: int) -> list:
    """Resize a list of (H, W, C) frames to (size, size, C) using nearest-neighbour."""
    return [
        np.array(Image.fromarray(f).resize((size, size), Image.NEAREST))
        for f in frames
    ]


# ---------------------------------------------------------------------------
# Clip 1 — 5×5 map, 3×3 obs window, 100 steps
# ---------------------------------------------------------------------------
print("Recording Clip 1: 5×5 map, 3×3 obs window, 100 steps …")

env = gym.make(
    "neo-gathering-v0",
    render_mode="rgb_array",
    map_size=(5, 5),
    obs_window=(3, 3),
)
env.reset(seed=42)
env_raw = env.unwrapped

frames_1 = [env.render()]
frames_1 += record_steps(env, env_raw, 50)
env.close()
print(f"  {len(frames_1)} frames collected")

# ---------------------------------------------------------------------------
# Clip 2 — 10×10 map, obs-window animation + 100 steps
# ---------------------------------------------------------------------------
print("Recording Clip 2: 10×10 map, obs-window animation + 100 steps …")

env = gym.make(
    "neo-gathering-v0",
    render_mode="rgb_array",
    map_size=(10, 10),
    obs_window=(3, 3),
    num_gold=10,
    num_silver=5,
    num_dragons=4,
)
env.reset(seed=8)
env_raw = env.unwrapped

frames_2 = []

# Pause on initial state (3×3)
frames_2 += [env.render()] * PAUSE_FRAMES
frames_2 += record_steps(env, env_raw, 10)

# Animate obs window
for size in (3, 5, 7):
    set_obs_window(env_raw, size)
    frames_2 += [env.render()] * PAUSE_FRAMES
    # frames_2 += record_steps(env, env_raw, 15)
    # frames_2 += [env.render()] * PAUSE_FRAMES

frames_2 += record_steps(env, env_raw, 50)
env.close()
print(f"  {len(frames_2)} frames collected")

# ---------------------------------------------------------------------------
# Stitch and write GIF
# ---------------------------------------------------------------------------
target_size = max(frames_1[0].shape[0], frames_2[0].shape[0])

if frames_1[0].shape[0] != target_size:
    frames_1 = resize_frames(frames_1, target_size)
if frames_2[0].shape[0] != target_size:
    frames_2 = resize_frames(frames_2, target_size)

gap = [np.zeros_like(frames_1[0])] * GAP_FRAMES
all_frames = frames_1 + gap + frames_2

out_path = os.path.join(OUTPUT_DIR, "neo_gathering.gif")
imageio.mimwrite(out_path, all_frames, fps=FPS, loop=0)
print(f"\n→ {out_path}  ({len(all_frames)} frames, {len(all_frames) / FPS:.1f} s)")

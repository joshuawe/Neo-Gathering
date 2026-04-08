"""
Play the NeoGathering environment interactively using arrow keys or WASD.

Controls:
    Arrow Up  / W  -> Move Up
    Arrow Down / S -> Move Down
    Arrow Left / A -> Move Left
    Arrow Right / D -> Move Right
    Q / Escape     -> Quit
"""

import gymnasium as gym
from gymnasium.utils.play import play

import neo_gathering  # noqa: F401 — registers the environment

keys_to_action = {
    # Arrow keys
    "u": 0,  # pygame UP
    "d": 1,  # pygame DOWN
    "l": 2,  # pygame LEFT
    "r": 3,  # pygame RIGHT
    # WASD
    "w": 0,
    "s": 1,
    "a": 2,
    "d": 3,
}

# pygame key constants (integers) are more reliable for special keys
# UP=273, DOWN=274, LEFT=276, RIGHT=275
keys_to_action_pg = {
    (273,): 0,  # Up arrow
    (274,): 1,  # Down arrow
    (276,): 2,  # Left arrow
    (275,): 3,  # Right arrow
    (ord("w"),): 0,
    (ord("s"),): 1,
    (ord("a"),): 2,
    (ord("d"),): 3,
}

env = gym.make("neo-gathering-v0", render_mode="rgb_array")

step_count = 0
total_reward = 0.0
action_names = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}


def on_step(*args):
    _, _, action, rew, terminated, truncated, _ = args
    global step_count, total_reward
    step_count += 1
    total_reward += rew
    pos = env.unwrapped.current_pos
    action_str = action_names.get(int(action), "?")
    print(
        f"\r  Step: {step_count:4d} | Pos: ({pos[0]:2d}, {pos[1]:2d})"
        f" | Action: {action_str:<5}"
        f" | Reward: {rew:+.1f} | Total: {total_reward:+.1f}   ",
        end="",
        flush=True,
    )
    if terminated or truncated:
        print(
            f"\n  Episode ended after {step_count} steps."
            f" Total reward: {total_reward:+.1f}"
        )
        step_count = 0
        total_reward = 0.0


print(__doc__)
print("Starting NeoGathering. Close the pygame window or press Q/Escape to quit.\n")

play(
    env,
    keys_to_action=keys_to_action_pg,
    noop=0,  # default action when no key pressed: move up
    zoom=2.0,
    wait_on_player=True,
    seed=42,
    callback=on_step,
)

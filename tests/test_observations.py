import numpy as np
import pytest

from neo_gathering.neo_gathering import NeoGathering

# map_size, window_size
WINDOW_SIZE_CASES = [
    ((5, 5), (3, 3)),
    ((5,5), (11,11)),
    ((7, 7), (5, 5)),
    ((10, 8), (3, 5)),
    ((5, 5), (1, 1)),
    ((6, 6), (3, 3)),
    ((8, 8), (7, 7)),   # window larger than map
]


@pytest.mark.parametrize("map_size,obs_window", WINDOW_SIZE_CASES)
def test_observation_window_size(map_size, obs_window):
    """Check that the observation window size is identical to desired one."""
    env = NeoGathering(map_size=map_size, obs_window=obs_window)
    env.reset()
    obs = env.get_observation()
    assert obs.shape == obs_window


WINDOW_CONTENT_CASES = [
    ((7, 7), (3, 3)),
    ((9, 9), (5, 5)),
    ((8, 10), (3, 5)),
]


@pytest.mark.parametrize("map_size,obs_window", WINDOW_CONTENT_CASES)
def test_observation_window_content(map_size, obs_window):
    """For interior agent position (no padding needed), window content matches map."""
    env = NeoGathering(map_size=map_size, obs_window=obs_window)
    env.reset()

    # Place agent at center, far enough from all edges that no padding is needed
    cx, cy = map_size[0] // 2, map_size[1] // 2
    env.current_pos = np.array([cx, cy])

    dx = obs_window[0] // 2
    dy = obs_window[1] // 2
    # Expected slice from the real map (no padding involved at center)
    expected = env.map[
        cx - dx : cx - dx + obs_window[0],
        cy - dy : cy - dy + obs_window[1],
    ]

    obs = env.get_observation()
    np.testing.assert_array_equal(obs, expected)


def test_observation_contains_agent_cell():
    """The cell at the center of the observation (for an odd window) must equal the map value at agent position."""
    env = NeoGathering(map_size=(7, 7), obs_window=(3, 3))
    env.reset()
    cx, cy = 3, 3
    env.current_pos = np.array([cx, cy])
    obs = env.get_observation()
    assert obs[1, 1] == env.map[cx, cy]


def test_observation_is_within_observation_space():
    """Observations must lie within the declared observation_space bounds."""
    env = NeoGathering(map_size=(7, 7), obs_window=(5, 5))
    env.reset()
    obs = env.get_observation()
    assert env.observation_space.contains(obs)


# Each entry: (map_size, obs_window, agent_pos)
# Covers corners, edges, and interior to ensure padding value (-1 = wall) appears exactly where expected.
PADDING_CASES = [
    ((5, 5), (3, 3), (0, 0), "top-left corner"),
    ((5, 5), (3, 3), (0, 4), "top-right corner"),
    ((5, 5), (3, 3), (4, 0), "bottom-left corner"),
    ((5, 5), (3, 3), (4, 4), "bottom-right corner"),
    ((5, 5), (3, 3), (0, 2), "top edge"),
    ((5, 5), (3, 3), (4, 2), "bottom edge"),
    ((5, 5), (3, 3), (2, 0), "left edge"),
    ((5, 5), (3, 3), (2, 4), "right edge"),
    ((5, 5), (3, 3), (2, 2), "interior — no padding"),
    ((5, 5), (5, 5), (0, 0), "5x5 window at top-left corner"),
    ((5, 5), (5, 5), (4, 4), "5x5 window at bottom-right corner"),
    ((5, 5), (31, 31), (2, 2), "window larger than map"),
]


@pytest.mark.parametrize("map_size,obs_window,agent_pos,description", PADDING_CASES)
def test_padding(map_size, obs_window, agent_pos, description):
    """Padding with wall value (-1) fills cells outside the map boundary."""
    env = NeoGathering(map_size=map_size, obs_window=obs_window)
    env.reset()

    wall = env.object_dict["wall"]  # -1
    env.current_pos = np.array(agent_pos)
    obs = env.get_observation()

    assert obs.shape == obs_window, f"[{description}] Wrong shape"

    x, y = agent_pos
    ws_x, ws_y = obs_window
    dx, dy = ws_x // 2, ws_y // 2

    # Verify cell by cell: out-of-map cells must be wall, in-map cells must match the map
    row_start_in_map = x - dx
    col_start_in_map = y - dy
    for obs_r in range(ws_x):
        map_r = row_start_in_map + obs_r
        for obs_c in range(ws_y):
            map_c = col_start_in_map + obs_c
            out_of_bounds = (
                map_r < 0 or map_r >= map_size[0]
                or map_c < 0 or map_c >= map_size[1]
            )
            if out_of_bounds:
                assert obs[obs_r, obs_c] == wall, (
                    f"[{description}] obs[{obs_r},{obs_c}] (map[{map_r},{map_c}]) "
                    f"expected wall ({wall}), got {obs[obs_r, obs_c]}"
                )
            else:
                assert obs[obs_r, obs_c] == env.map[map_r, map_c], (
                    f"[{description}] obs[{obs_r},{obs_c}] (map[{map_r},{map_c}]) "
                    f"expected {env.map[map_r, map_c]}, got {obs[obs_r, obs_c]}"
                )

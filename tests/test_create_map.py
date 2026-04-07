import pytest
import numpy as np

from neo_gathering.neo_gathering import NeoGathering


# Parameterized cases covering various map sizes and item counts.
MAP_CASES = [
    ((5, 5), 1, 1, 2),   # default-style params
    ((3, 4), 1, 1, 1),   # non-square map
    ((10, 7), 3, 2, 4),  # larger map with more items
    ((2, 5), 1, 1, 1),   # narrow map, minimal items
]


@pytest.mark.parametrize("map_size,num_gold,num_silver,num_dragons", MAP_CASES)
def test_map_has_correct_shape(map_size, num_gold, num_silver, num_dragons):
    env = NeoGathering(map_size=map_size, num_gold=num_gold, num_silver=num_silver, num_dragons=num_dragons)
    env.reset()
    assert env.map.shape == map_size


@pytest.mark.parametrize("map_size,num_gold,num_silver,num_dragons", MAP_CASES)
def test_map_has_exactly_one_home(map_size, num_gold, num_silver, num_dragons):
    env = NeoGathering(map_size=map_size, num_gold=num_gold, num_silver=num_silver, num_dragons=num_dragons)
    env.reset()
    assert np.sum(env.map == env.object_dict["home"]) == 1


@pytest.mark.parametrize("map_size,num_gold,num_silver,num_dragons", MAP_CASES)
def test_map_has_correct_number_of_gold(map_size, num_gold, num_silver, num_dragons):
    env = NeoGathering(map_size=map_size, num_gold=num_gold, num_silver=num_silver, num_dragons=num_dragons)
    env.reset()
    assert np.sum(env.map == env.object_dict["gold"]) == num_gold


@pytest.mark.parametrize("map_size,num_gold,num_silver,num_dragons", MAP_CASES)
def test_map_has_correct_number_of_silver(map_size, num_gold, num_silver, num_dragons):
    env = NeoGathering(map_size=map_size, num_gold=num_gold, num_silver=num_silver, num_dragons=num_dragons)
    env.reset()
    assert np.sum(env.map == env.object_dict["silver"]) == num_silver


@pytest.mark.parametrize("map_size,num_gold,num_silver,num_dragons", MAP_CASES)
def test_map_has_correct_number_of_dragons(map_size, num_gold, num_silver, num_dragons):
    env = NeoGathering(map_size=map_size, num_gold=num_gold, num_silver=num_silver, num_dragons=num_dragons)
    env.reset()
    assert np.sum(env.map == env.object_dict["dragon"]) == num_dragons


def test_all_items_placed_on_distinct_cells():
    """No cell should contain more than one item."""
    env = NeoGathering(map_size=(5, 5), num_gold=1, num_silver=1, num_dragons=2)
    env.reset()
    non_empty = env.map[env.map != 0]
    expected_count = 1 + 1 + 1 + 2  # home + gold + silver + dragons
    assert len(non_empty) == expected_count


MOVEMENT_CASES = [
    (5, 5),
    (3, 7),
    (10, 4),
]


@pytest.mark.parametrize("rows,cols", MOVEMENT_CASES)
def test_movements(rows, cols):
    """
    For different map sizes, verify that every cell is reachable by walking
    a systematic path and that wall-collisions (edges) are enforced.
    See `NeoGathering` for integer values to action mapping.
    """
    env = NeoGathering(map_size=(rows, cols))
    env.reset()
    
    UP = env.action_dict['up']
    DOWN = env.action_dict['down']
    LEFT = env.action_dict['left']
    RIGHT = env.action_dict['right']

    # ---- 1. reach every cell by sweeping row by row ----
    visited = set()

    # move to top-left corner
    for _ in range(rows):
        env.step(UP)
    for _ in range(cols):
        env.step(LEFT)

    for row in range(rows):
        for col in range(cols):
            pos = tuple(env.current_pos)
            visited.add(pos)
            if col < cols - 1:
                env.step(RIGHT)
        # move down one row and reverse direction
        env.step(DOWN)
        if row < rows - 1:
            for _ in range(cols - 1):
                env.step(LEFT)

    all_cells = {(r, c) for r in range(rows) for c in range(cols)}
    assert visited == all_cells, (
        f"Not all cells visited. Missing: {all_cells - visited}"
    )

    # ---- 2. wall-collision: stepping beyond an edge must not move the agent ----
    # drive to top-left corner
    for _ in range(rows):
        env.step(UP)
    for _ in range(cols):
        env.step(LEFT)

    assert tuple(env.current_pos) == (0, 0)

    env.step(UP)    # already at top row — must stay
    assert tuple(env.current_pos) == (0, 0), "UP from row 0 should be blocked"

    env.step(LEFT)  # already at left col — must stay
    assert tuple(env.current_pos) == (0, 0), (
        "LEFT from col 0 should be blocked"
    )

    # drive to bottom-right corner
    for _ in range(rows):
        env.step(DOWN)
    for _ in range(cols):
        env.step(RIGHT)

    assert tuple(env.current_pos) == (rows - 1, cols - 1)

    env.step(DOWN)   # already at bottom row — must stay
    assert tuple(env.current_pos) == (rows - 1, cols - 1), (
        "DOWN from last row should be blocked"
    )
    env.step(RIGHT)  # already at right col — must stay
    assert tuple(env.current_pos) == (rows - 1, cols - 1), (
        "RIGHT from last col should be blocked"
    )
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
    assert env.map.shape == map_size


@pytest.mark.parametrize("map_size,num_gold,num_silver,num_dragons", MAP_CASES)
def test_map_has_exactly_one_home(map_size, num_gold, num_silver, num_dragons):
    env = NeoGathering(map_size=map_size, num_gold=num_gold, num_silver=num_silver, num_dragons=num_dragons)
    assert np.sum(env.map == env.object_dict["home"]) == 1


@pytest.mark.parametrize("map_size,num_gold,num_silver,num_dragons", MAP_CASES)
def test_map_has_correct_number_of_gold(map_size, num_gold, num_silver, num_dragons):
    env = NeoGathering(map_size=map_size, num_gold=num_gold, num_silver=num_silver, num_dragons=num_dragons)
    assert np.sum(env.map == env.object_dict["gold"]) == num_gold


@pytest.mark.parametrize("map_size,num_gold,num_silver,num_dragons", MAP_CASES)
def test_map_has_correct_number_of_silver(map_size, num_gold, num_silver, num_dragons):
    env = NeoGathering(map_size=map_size, num_gold=num_gold, num_silver=num_silver, num_dragons=num_dragons)
    assert np.sum(env.map == env.object_dict["silver"]) == num_silver


@pytest.mark.parametrize("map_size,num_gold,num_silver,num_dragons", MAP_CASES)
def test_map_has_correct_number_of_dragons(map_size, num_gold, num_silver, num_dragons):
    env = NeoGathering(map_size=map_size, num_gold=num_gold, num_silver=num_silver, num_dragons=num_dragons)
    assert np.sum(env.map == env.object_dict["dragon"]) == num_dragons


def test_all_items_placed_on_distinct_cells():
    """No cell should contain more than one item."""
    env = NeoGathering(map_size=(5, 5), num_gold=1, num_silver=1, num_dragons=2)
    non_empty = env.map[env.map != 0]
    expected_count = 1 + 1 + 1 + 2  # home + gold + silver + dragons
    assert len(non_empty) == expected_count

import numpy as np
import pytest

from neo_gathering.neo_gathering import NeoGathering


# Object cell values matching env.object_dict
H, G, S, D = 1, 3, 4, 2  # home, gold, silver/gem, dragon


def make_env_with_map(grid: np.ndarray) -> NeoGathering:
    """Construct a NeoGathering env with the exact map provided."""
    rows, cols = grid.shape
    env = NeoGathering(
        map_size=(rows, cols),
        num_gold=int(np.sum(grid == G)),
        num_silver=int(np.sum(grid == S)),
        num_dragons=int(np.sum(grid == D)),
    )
    env.reset(seed=0)
    env.map = grid.copy()
    env.current_pos = env._get_home_position()
    env.has_all_gold = 0
    env.has_all_diamond = 0
    return env


# ---------------------------------------------------------------------------
# Hand-crafted maps
# ---------------------------------------------------------------------------

# Map A – linear layout, no dragons.
# H G S . .
# . . . . .
# . . . . .
# . . . . .
# . . . . .
# H=(0,0), G=(0,1), S=(0,2)
# Optimal: H->G (1 step), G->S (1 step), S->G->H (2 steps)
# path = [(0,0),(0,1),(0,2),(0,1),(0,0)], len=5
MAP_LINEAR = np.array(
    [[H, G, S, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]],
    dtype=np.int16,
)

# Map B – dragon sits between home and gold; agent must go around via row 1.
# H D G S .
# . . . . .
# . . . . .
# . . . . .
# . . . . .
# H=(0,0), D=(0,1), G=(0,2), S=(0,3)
# path len=11
MAP_DRAGON_AVOIDANCE = np.array(
    [[H, D, G, S, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]],
    dtype=np.int16,
)

# Map C – dragons completely wall off home; no dragon-free path exists anywhere.
# H D G
# D D .
# S . .
# H=(0,0), G=(0,2), S=(2,0)
# Dragons at (0,1),(1,0),(1,1) – every neighbour of H is a dragon.
# path len=9
MAP_FORCED_DRAGON = np.array(
    [[H, D, G],
     [D, D, 0],
     [S, 0, 0]],
    dtype=np.int16,
)

# Map D – spread layout, no dragons, larger distances.
# H . . . .
# . . . . .
# . . G . .
# . . . . .
# . . . . S
# H=(0,0), G=(2,2), S=(4,4)
# path len=17
MAP_SPREAD = np.array(
    [[H, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, G, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, S]],
    dtype=np.int16,
)


# ---------------------------------------------------------------------------
# Structural / invariant tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid", [MAP_LINEAR, MAP_DRAGON_AVOIDANCE, MAP_SPREAD])
def test_path_starts_and_ends_at_home(grid):
    env = make_env_with_map(grid)
    home_pos = tuple(int(x) for x in env._get_home_position())
    path, rewards = env.shortest_path()
    assert len(path) > 0, "Path must be non-empty."
    assert path[0] == home_pos, "Path must start at home."
    assert path[-1] == home_pos, "Path must end at home."


@pytest.mark.parametrize("grid", [MAP_LINEAR, MAP_DRAGON_AVOIDANCE, MAP_SPREAD])
def test_rewards_length_matches_path(grid):
    env = make_env_with_map(grid)
    path, rewards = env.shortest_path()
    assert len(rewards) == len(path) - 1


@pytest.mark.parametrize("grid", [MAP_LINEAR, MAP_DRAGON_AVOIDANCE, MAP_SPREAD])
def test_path_visits_gold_before_gem(grid):
    env = make_env_with_map(grid)
    gold_positions = {
        tuple(int(x) for x in pos)
        for pos in np.argwhere(env.map == G)
    }
    silver_positions = {
        tuple(int(x) for x in pos)
        for pos in np.argwhere(env.map == S)
    }
    path, _ = env.shortest_path()

    gold_indices = [i for i, p in enumerate(path) if p in gold_positions]
    silver_indices = [i for i, p in enumerate(path) if p in silver_positions]

    assert gold_indices, "Gold must appear in path."
    assert silver_indices, "Gem must appear in path."
    # The first visit to gold must precede the first visit to gem
    assert min(gold_indices) < min(silver_indices), (
        "Gold must be collected before gem."
    )


@pytest.mark.parametrize("grid", [MAP_LINEAR, MAP_DRAGON_AVOIDANCE, MAP_SPREAD])
def test_gem_visited_before_final_home_return(grid):
    """Gem must appear in path before the final home cell."""
    env = make_env_with_map(grid)
    silver_positions = {
        tuple(int(x) for x in pos)
        for pos in np.argwhere(env.map == S)
    }
    path, _ = env.shortest_path()
    last_silver_idx = max(i for i, p in enumerate(path) if p in silver_positions)
    assert last_silver_idx < len(path) - 1, (
        "Gem must be visited before the final home cell."
    )


@pytest.mark.parametrize("grid", [MAP_LINEAR, MAP_DRAGON_AVOIDANCE, MAP_SPREAD])
def test_final_reward_is_two(grid):
    """Collecting both items and returning home yields total scalar reward of 2."""
    env = make_env_with_map(grid)
    _, rewards = env.shortest_path()
    assert rewards[-1] == pytest.approx(2.0), (
        "Last reward (home arrival with both items) must be 2.0."
    )


@pytest.mark.parametrize("grid", [MAP_LINEAR, MAP_DRAGON_AVOIDANCE, MAP_SPREAD])
def test_intermediate_rewards_are_zero_or_negative(grid):
    """All rewards before the last step must be 0.0 (or negative for dragons)."""
    env = make_env_with_map(grid)
    _, rewards = env.shortest_path()
    for r in rewards[:-1]:
        assert r <= 0.0, f"Intermediate reward {r} must be <= 0."


# ---------------------------------------------------------------------------
# Dragon avoidance tests
# ---------------------------------------------------------------------------

def test_path_avoids_dragon_when_possible():
    env = make_env_with_map(MAP_DRAGON_AVOIDANCE)
    dragon_positions = {
        tuple(int(x) for x in pos)
        for pos in np.argwhere(env.map == D)
    }
    path, _ = env.shortest_path()
    for cell in path:
        assert cell not in dragon_positions, (
            f"Path cell {cell} is a dragon – should have been avoided."
        )


def test_path_uses_dragon_cells_when_unavoidable():
    """When dragons wall off home on all sides, the path must pass through a dragon."""
    env = make_env_with_map(MAP_FORCED_DRAGON)
    dragon_positions = {
        tuple(int(x) for x in pos)
        for pos in np.argwhere(env.map == D)
    }
    path, rewards = env.shortest_path()
    assert len(path) > 0, "Fallback path must still be returned."
    assert any(cell in dragon_positions for cell in path), (
        "When no dragon-free path exists, path must go through a dragon cell."
    )


# ---------------------------------------------------------------------------
# Exact path length tests
# (Expected values verified by hand; correct if marked with # FIX ME)
# ---------------------------------------------------------------------------

def test_path_length_linear():
    # H G S in a row: H->G (1), G->S (1), S->G->H (2) = 4 moves = 5 cells
    env = make_env_with_map(MAP_LINEAR)
    path, _ = env.shortest_path()
    assert len(path) == 5


def test_path_length_spread():
    # H=(0,0), G=(2,2), S=(4,4)
    # H->G: manhattan 4 = 4 steps
    # G->S: manhattan 4 = 4 steps
    # S->H: manhattan 8 = 8 steps
    # Total cells = 4+4+8+1 = 17
    env = make_env_with_map(MAP_SPREAD)
    path, _ = env.shortest_path()
    assert len(path) == 17 


def test_path_length_dragon_avoidance():
    # H=(0,0) D=(0,1) G=(0,2) S=(0,3); must detour via row 1
    # H->G: (0,0)->(1,0)->(1,1)->(1,2)->(0,2) = 4 steps
    # G->S: (0,2)->(0,3) = 1 step
    # S->H: (0,3)->(1,3)->(1,2)->(1,1)->(1,0)->(0,0) = 5 steps
    # Total cells = 4+1+5+1 = 11
    env = make_env_with_map(MAP_DRAGON_AVOIDANCE)
    path, _ = env.shortest_path()
    assert len(path) == 11
    
def test_path_length_map_forced_drageon():
    env = make_env_with_map(MAP_FORCED_DRAGON)
    path, _ = env.shortest_path()
    assert len(path) == 9

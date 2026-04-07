import logging
from os import path
from typing import List, Optional

from itertools import product

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle


logger = logging.getLogger(__name__)

class NeoGathering(gym.Env, EzPickle):
    """
    ## Description
    Adaption from "Barrett, Leon & Narayanan, Srini. (2008). Learning all optimal policies with multiple criteria.
    Proceedings of the 25th International Conference on Machine Learning. 41-47. 10.1145/1390156.1390162."

    ## Observation Space
    The observation is discrete and consists of 4 elements:
    - 0: The x coordinate of the agent
    - 1: The y coordinate of the agent
    - 2: Flag indicating if the agent collected the gold
    - 3: Flag indicating if the agent collected the diamond

    ## Action Space
    The action is discrete and consists of 4 elements:
    - 0: Move up
    - 1: Move down
    - 2: Move left
    - 3: Move right

    ## Reward Space
    The reward is 3-dimensional:
    - 0: -1 if killed by an enemy, else 0
    - 1: +1 if returned home with gold, else 0
    - 2: +1 if returned home with diamond, else 0

    ## Starting State
    The agent starts at the home position with no gold or diamond.

    ## Episode Termination
    The episode terminates when the agent returns home, or when the agent is killed by an enemy.

    ## Credits
    The home asset is from https://limezu.itch.io/serenevillagerevamped
    The gold, enemy and gem assets are from https://ninjikin.itch.io/treasure
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_dragons: int = 2,
        num_gold: int = 1,
        num_silver: int = 1,
        map_size: tuple = (5, 5),
        obs_window: tuple = (3,3),
    ):
        EzPickle.__init__(self, render_mode)

        # type check
        assert isinstance(num_dragons, int), (
            f"num_dragons should be int, was {type(num_dragons)}"
        )
        assert isinstance(num_gold, int), (
            f"num_dragons should be int, was {type(num_gold)}"
        )
        assert isinstance(num_silver, int), (
            f"num_dragons should be int, was {type(num_silver)}"
        )
        assert len(map_size) == 2, (
            f"Invalid map_size. Expected 2 dims, got {len(map_size)}."
        )
        assert isinstance(map_size[0], int) and isinstance(map_size[1], int), (
            f"map_size should have two ints, got {type(map_size[0])} and {type(map_size[1])}"
        )
        assert (
            np.sum([num_dragons, num_gold, num_silver, 1]) <= map_size[0] * map_size[1]
        ), f"map_size is too small for the desired number of items."
        # assert map_size[0]>obs_window[0] and map_size[1] > obs_window[1] # TODO: should we allow observation windows larger than map? Do allow larger than 2x the size?
        if (obs_window[0]%2 == 0) or (obs_window[1]%2 == 0):
            logger.warning("At least one dim of the observation window is even, this results in the viewing box being NOT centered around the agent. In order to center the agent perfectly in the observation window use uneven numbers.")  # noqa: E501

        self.render_mode = render_mode

        self.map_size = map_size
        self.obs_window = obs_window

        self.object_dict = {"wall": -1, "home": 1, "dragon": 2, "gold": 3, "silver": 4}
        self.num_dragons = num_dragons
        self.num_gold = num_gold
        self.num_silver = num_silver

        self.map = None
        self._padded_map = None  # the map with padding around
        self._ws_x = self.obs_window[0] # window size in x-dim
        self._ws_y = self.obs_window[1] # window size in y-dim
        self._dx = self.obs_window[0] // 2  # used for computing window bounds 
        self._dy = self.obs_window[1] // 2  # used for computing window bounds
        self.obs = np.zeros(self.obs_window, dtype=np.int16) # cached array for obs
        self.current_pos = None

        self.action_dict = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.direction_dict = {
            0: np.array([-1, 0], dtype=np.int32),  # up
            1: np.array([1, 0], dtype=np.int32),  # down
            2: np.array([0, -1], dtype=np.int32),  # left
            3: np.array([0, 1], dtype=np.int32),  # right
        }

        self.observation_space = Box(
            low=min(self.object_dict.values()),
            high=max(self.object_dict.values()),
            shape=self.obs_window,
            dtype=np.int16,
        )

        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.action_space = Discrete(4)
        # reward space:
        self.reward_space = Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([0.0, 1.0, 1.0]),
            shape=(3,),
            dtype=np.float32,
        )
        self.reward_dim = 3

        # pygame
        self.size = 5
        self.cell_size = (64, 64)
        self.window_size = (
            self.map_size[1] * self.cell_size[1],
            self.map_size[0] * self.cell_size[0],
        )
        self.clock = None
        self.elf_images = []
        self.gold_img = None
        self.gem_img = None
        self.enemy_img = None
        self.home_img = None
        self.mountain_bg_img = []
        self.window = None
        self.last_action = None

    def create_map(self, map_size: tuple = None) -> np.ndarray:
        map_size = self.map_size if map_size is None else map_size
        # empty map
        map = np.zeros(map_size, dtype=np.int16)
        # all possible map/cell positions
        options = product(np.arange(map_size[0]), np.arange(map_size[1]))
        options = list(options)
        # pick positions: Home, Dragon 1, Dragon 2, Gold, Sivler
        num_picks = 1 + self.num_dragons + self.num_gold + self.num_silver
        picks: list = self.np_random.choice(
            range(len(options)), size=num_picks, replace=False
        ).tolist()
        positions = [options[i] for i in picks]
        # populate the map
        map[positions.pop()] = self.object_dict["home"]
        for i in range(self.num_dragons):
            map[positions.pop()] = self.object_dict["dragon"]
        for i in range(self.num_gold):
            map[positions.pop()] = self.object_dict["gold"]
        for i in range(self.num_silver):
            map[positions.pop()] = self.object_dict["silver"]

        return map

    def get_map_value(self, pos):
        return self.map[pos[0]][pos[1]]

    def is_valid_observation(self, observation):
        return (
            observation[0] >= 0
            and observation[0] < self.map_size[0]
            and observation[1] >= 0
            and observation[1] < self.map_size[1]
        )
    
    def get_observation(self):
        x, y = self.current_pos
        self.obs[:] = self._padded_map[x : x + self._ws_x, y : y + self._ws_y]
        return self.obs

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        # create map
        self.map = self.create_map(self.map_size)
        # precompute the padded map once
        self._padded_map = np.pad(
            self.map, ((self._dx, self._dx), (self._dy, self._dy)),
            constant_values=self.object_dict["wall"],
        )
        self.current_pos = self._get_home_position()

        self.has_gem = 0
        self.has_gold = 0
        self.step_count = 0.0
        observation = self.get_observation()
        if self.render_mode == "human":
            self.render()
        return observation, {}

    def step(self, action):
        action = int(action)
        next_pos = self.current_pos + self.direction_dict[action]
        self.last_action = action

        if self.is_valid_observation(next_pos):
            self.current_pos = next_pos

        vec_reward, done = self.get_reward()

        observation = self.get_observation()
        if self.render_mode == "human":
            self.render()
        return observation, vec_reward, done, False, {}

    def get_reward(self):
        vec_reward = np.zeros(3, dtype=np.float32)
        done = False

        cell = self.get_map_value(self.current_pos)
        self.has_gold |= cell == self.object_dict["gold"]
        self.has_gem |= cell == self.object_dict["silver"]
        if cell == self.object_dict["dragon"]:
            # 0.9 chance of survival, 0.1 chance of death
            if self.np_random.random() < 0.1:
                vec_reward[0] = -1.0
                done = True
        elif cell == self.object_dict["home"]:
            # if you are home again
            done = True
            vec_reward[1] = self.has_gold
            vec_reward[2] = self.has_gem
        reward = np.sum(vec_reward).item()
        return reward, done

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def _get_home_position(self) -> tuple[int, int]:
        wheres = np.argwhere(self.map == self.object_dict["home"])
        assert len(wheres) == 1, (
            f"Found to many 'home' cells. Should be 1, found {len(wheres)}"
        )
        position = wheres[0]
        return position
    
    def shortest_path(self) -> tuple[list, list]:
        """
        Compute the optimal (shortest) path: home -> gold -> gem -> home.

        Dragons are treated as impassable. If no dragon-free path exists,
        falls back to allowing dragon cells (using expected reward -0.1 per cell).

        Returns
        -------
        path : list of (row, col) tuples
            Every cell visited, starting and ending at home.
        rewards : list of float
            Scalar reward received when stepping into each cell after the start.
            len(rewards) == len(path) - 1.
        """
        import heapq

        assert self.map is not None, "Call env.reset() before shortest_path()."

        home_pos = tuple(int(x) for x in self._get_home_position())
        gold_positions = [
            tuple(int(x) for x in pos)
            for pos in np.argwhere(self.map == self.object_dict["gold"])
        ]
        silver_positions = [
            tuple(int(x) for x in pos)
            for pos in np.argwhere(self.map == self.object_dict["silver"])
        ]
        dragon_positions = {
            tuple(int(x) for x in pos)
            for pos in np.argwhere(self.map == self.object_dict["dragon"])
        }

        assert gold_positions, "No gold on the map."
        assert silver_positions, "No gem/silver on the map."

        def heuristic(a: tuple, b: tuple) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def astar(start: tuple, goal: tuple, avoid: set) -> list | None:
            if start == goal:
                return [start]
            # heap entries: (f, g, pos) — no path stored in heap
            heap = [(heuristic(start, goal), 0, start)]
            came_from = {start: None}
            while heap:
                _, g, pos = heapq.heappop(heap)
                for delta in self.direction_dict.values():
                    nxt = (pos[0] + int(delta[0]), pos[1] + int(delta[1]))
                    if not self.is_valid_observation(nxt) or nxt in came_from or nxt in avoid:  # noqa: E501
                        continue
                    came_from[nxt] = pos
                    if nxt == goal:
                        # reconstruct path by backtracking through came_from
                        path = []
                        node = nxt
                        while node is not None:
                            path.append(node)
                            node = came_from[node]
                        path.reverse()
                        return path
                    ng = g + 1
                    heapq.heappush(heap, (ng + heuristic(nxt, goal), ng, nxt))
            return None

        best_path = None
        for gold_pos in gold_positions:
            for silver_pos in silver_positions:
                for avoid in (dragon_positions, set()):  # prefer dragon-free
                    p1 = astar(home_pos, gold_pos, avoid)
                    p2 = astar(gold_pos, silver_pos, avoid) if p1 is not None else None
                    p3 = astar(silver_pos, home_pos, avoid) if p2 is not None else None
                    if p1 is not None and p2 is not None and p3 is not None:
                        full_path = p1 + p2[1:] + p3[1:]
                        if best_path is None or len(full_path) < len(best_path):
                            best_path = full_path
                        break  # dragon-free path found for this gold/silver combo

        if best_path is None:
            return [], []

        # Simulate rewards: rewards[i] = reward for stepping into path[i+1]
        has_gold, has_gem = 0, 0
        rewards = []
        for pos in best_path[1:]:
            cell = self.map[pos[0]][pos[1]]
            if cell == self.object_dict["gold"]:
                has_gold = 1
                reward = 0.0
            elif cell == self.object_dict["silver"]:
                has_gem = 1
                reward = 0.0
            elif cell == self.object_dict["dragon"]:
                reward = -0.1  # expected value: 10% chance of -1
            elif cell == self.object_dict["home"]:
                reward = float(has_gold + has_gem)
            else:
                reward = 0.0
            rewards.append(reward)

        return best_path, rewards

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. mo_gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.window is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Resource Gathering")
                self.window = pygame.display.set_mode(self.window_size)
            else:
                self.window = pygame.Surface(self.window_size)

            if self.clock is None:
                self.clock = pygame.time.Clock()

            if not self.elf_images:
                hikers = [
                    path.join(path.dirname(__file__), "assets/elf_up.png"),
                    path.join(path.dirname(__file__), "assets/elf_down.png"),
                    path.join(path.dirname(__file__), "assets/elf_left.png"),
                    path.join(path.dirname(__file__), "assets/elf_right.png"),
                ]
                self.elf_images = [
                    pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                    for f_name in hikers
                ]
            if not self.mountain_bg_img:
                bg_imgs = [
                    path.join(path.dirname(__file__), "assets/mountain_bg1.png"),
                    path.join(path.dirname(__file__), "assets/mountain_bg2.png"),
                ]
                self.mountain_bg_img = [
                    pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                    for f_name in bg_imgs
                ]
            if self.gold_img is None:
                self.gold_img = pygame.transform.scale(
                    pygame.image.load(
                        path.join(path.dirname(__file__), "assets/gold.png")
                    ),
                    (0.6 * self.cell_size[0], 0.6 * self.cell_size[1]),
                )
            if self.gem_img is None:
                self.gem_img = pygame.transform.scale(
                    pygame.image.load(
                        path.join(path.dirname(__file__), "assets/gem.png")
                    ),
                    (0.6 * self.cell_size[0], 0.6 * self.cell_size[1]),
                )
            if self.enemy_img is None:
                self.enemy_img = pygame.transform.scale(
                    pygame.image.load(
                        path.join(path.dirname(__file__), "assets/enemy.png")
                    ),
                    (0.8 * self.cell_size[0], 0.8 * self.cell_size[1]),
                )
            if self.home_img is None:
                self.home_img = pygame.transform.scale(
                    pygame.image.load(
                        path.join(path.dirname(__file__), "assets/home.png")
                    ),
                    self.cell_size,
                )

        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                check_board_mask = i % 2 ^ j % 2
                self.window.blit(
                    self.mountain_bg_img[check_board_mask],
                    np.array([j, i]) * self.cell_size[0],
                )
                if self.map[i, j] == self.object_dict["gold"] and not self.has_gold:
                    self.window.blit(
                        self.gold_img,
                        np.array([j + 0.22, i + 0.25]) * self.cell_size[0],
                    )
                elif self.map[i, j] == self.object_dict["silver"] and not self.has_gem:
                    self.window.blit(
                        self.gem_img, np.array([j + 0.22, i + 0.25]) * self.cell_size[0]
                    )
                elif self.map[i, j] == self.object_dict["dragon"]:
                    self.window.blit(
                        self.enemy_img, np.array([j + 0.1, i + 0.1]) * self.cell_size[0]
                    )
                elif self.map[i, j] == self.object_dict["home"]:
                    self.window.blit(
                        self.home_img, np.array([j, i]) * self.cell_size[0]
                    )
        # Draw observation window overlay
        x, y = self.current_pos
        ws_x, ws_y = self.obs_window
        dx, dy = ws_x // 2, ws_y // 2
        row_start = max(0, x - dx)
        row_end = min(self.map.shape[0], x - dx + ws_x)
        col_start = max(0, y - dy)
        col_end = min(self.map.shape[1], y - dy + ws_y)
        pw = (col_end - col_start) * self.cell_size[0]
        ph = (row_end - row_start) * self.cell_size[1]
        obs_surface = pygame.Surface((pw, ph), pygame.SRCALPHA)
        obs_surface.fill((60, 130, 255, 18))
        pygame.draw.rect(
            obs_surface, (80, 150, 255, 200), obs_surface.get_rect(), width=2
        )
        self.window.blit(
            obs_surface, (col_start * self.cell_size[0], row_start * self.cell_size[1])
        )

        last_action = self.last_action if self.last_action is not None else 2
        self.window.blit(
            self.elf_images[last_action], self.current_pos[::-1] * self.cell_size[0]
        )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )


if __name__ == "__main__":
    import mo_gymnasium as mo_gym

    env = mo_gym.make("resource-gathering-v0", render_mode="human")
    terminated = False
    env.reset()
    while True:
        env.render()
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()

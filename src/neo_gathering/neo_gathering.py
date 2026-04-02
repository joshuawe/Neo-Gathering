from os import path
from typing import List, Optional

from itertools import product

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle


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

    def __init__(self, render_mode: Optional[str] = None):
        EzPickle.__init__(self, render_mode)

        self.render_mode = render_mode
        
        self.map_size = (5,5)

        self.object_dict = {"home": 1, "dragon": 2, "gold": 3, "silver": 4}
        self.num_dragons = 2
        self.num_gold = 1
        self.num_silver = 1

        # The map of resource gathering env
        # self.map = np.array(
        #     [
        #         [" ", " ", "R1", "E2", " "],
        #         [" ", " ", "E1", " ", "R2"],
        #         [" ", "H", " ", " ", " "],
        #         [" ", " ", " ", "E1", " "],
        #         [" ", " ", "H", " ", " "],
        #     ]
        # )
        self.map = self.create_map()
        self.initial_pos = np.array([4, 2], dtype=np.int32)

        self.dir = {
            0: np.array([-1, 0], dtype=np.int32),  # up
            1: np.array([1, 0], dtype=np.int32),  # down
            2: np.array([0, -1], dtype=np.int32),  # left
            3: np.array([0, 1], dtype=np.int32),  # right
        }

        self.observation_space = Box(low=0.0, high=5.0, shape=(4,), dtype=np.int32)

        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.action_space = Discrete(4)
        # reward space:
        self.reward_space = Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([0.0, 1.0, 1.0]),
            shape=(3,),
            dtype=np.float64,
        )
        self.reward_dim = 3

        # pygame
        self.size = 5
        self.cell_size = (64, 64)
        self.window_size = (
            self.map.shape[1] * self.cell_size[1],
            self.map.shape[0] * self.cell_size[0],
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

    def create_map(self, map_size: Optional[tuple] = None, seed: Optional[int] = None) -> np.ndarray:
        map_size = self.map_size if map_size is None else map_size
        # empty map
        map = np.zeros(map_size, dtype=np.int16)
        # all possible map/cell positions
        options = product(np.arange(map_size[0]), np.arange(map_size[1]))
        options = list(options)
        # pick positions: Home, Dragon 1, Dragon 2, Gold, Sivler
        num_picks = 1 + self.num_dragons + self.num_gold + self.num_silver
        picks: list = np.random.choice(range(len(options)), size=num_picks, replace=False).tolist()
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
            and observation[0] < self.size
            and observation[1] >= 0
            and observation[1] < self.size
        )

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

    def get_observation(self):
        pos = self.current_pos.copy()
        observation = np.concatenate(
            (pos, np.array([self.has_gold, self.has_gem], dtype=np.int32))
        )
        return observation

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        self.current_pos = self.initial_pos
        self.has_gem = 0
        self.has_gold = 0
        self.step_count = 0.0
        observation = self.get_observation()
        if self.render_mode == "human":
            self.render()
        return observation, {}

    def step(self, action):
        action = int(action)
        next_pos = self.current_pos + self.dir[action]
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
        if cell == self.object_dict["gold"]:
            self.has_gold = 1
        elif cell == self.object_dict["silver"]:
            self.has_gem = 1
        elif cell == self.object_dict["dragon"]:
            # 0.9 chance of survival, 0.1 chance of death
            if self.np_random.random() < 0.1:
                vec_reward[0] = -1.0
                done = True
        elif cell == self.object_dict["home"]:
            # if you are home again
            done = True
            vec_reward[1] = self.has_gold
            vec_reward[2] = self.has_gem
        return vec_reward,done

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


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

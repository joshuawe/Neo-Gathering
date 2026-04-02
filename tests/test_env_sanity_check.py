import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import neo_gathering

def test_load_resource_gathering():
    env = gym.make("resource-gathering-v0", render_mode="rgb_array")
    env.reset()
    env.step(env.action_space.sample())
    
    env = gym.make("resource-gathering-v0", render_mode="human")
    env.reset()
    env.step(env.action_space.sample())
    
    
def test_load_neo_gathering():
    env = gym.make("neo-gathering-v0", render_mode="rgb_array")
    env.reset()
    env.step(env.action_space.sample())
    
    env = gym.make("neo-gathering-v0", render_mode="human")
    env.reset()
    env.step(env.action_space.sample())

def test_sanity_resource_gathering():
    env = gym.make("resource-gathering-v0", render_mode="rgb_array")
    check_env(env=env.unwrapped)
    
def test_sanity_neo_gathering():
    env = gym.make("neo-gathering-v0", render_mode="rgb_array")
    check_env(env=env.unwrapped)
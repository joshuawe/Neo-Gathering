from gymnasium.envs.registration import register


register(
    id="neo-gathering-v0",
    entry_point="neo_gathering.neo_gathering:NeoGathering",
    max_episode_steps=100,
)

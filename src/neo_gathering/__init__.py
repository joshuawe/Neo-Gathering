from gymnasium.envs.registration import register


register(
    id="neo-gathering-v0",
    entry_point="neo_gathering.neo_gathering:NeoGathering",
    max_episode_steps=1000,
)

register(
    id="resource-gathering-v0",
    entry_point="neo_gathering.resource_gathering:ResourceGathering",
    max_episode_steps=100,
)

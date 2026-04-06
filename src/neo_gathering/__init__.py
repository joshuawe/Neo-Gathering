from gymnasium.envs.registration import register


register(
    id="neo-gathering-v0",
    entry_point="neo_gathering.neo_gathering:NeoGathering",
    max_episode_steps=100, # TODO: do we want a fixed max_episode?
)

register(
    id="resource-gathering-v0",
    entry_point="neo_gathering.resource_gathering:ResourceGathering",
    max_episode_steps=100,
)

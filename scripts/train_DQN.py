import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# import sbx
import stable_baselines3 as sb3

# from gymnasium.wrappers import FrameStackObservation
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from tqdm.auto import tqdm

import neo_gathering

# from neo_gathering.neo_gathering import NeoGathering
# from neo_gathering.resource_gathering import ResourceGathering


N_EVAL_EPISODES = 100
N_TRAIN_STEPS = int(10e5)
SEED = 442

N_ENVS = 12

env_kwargs = {"render_mode": None, "obs_window": (11, 11)}
wrapper_kwargs = {"n_stack": 10}
env = make_vec_env(
    "neo-gathering-v0",
    env_kwargs=env_kwargs,
    n_envs=N_ENVS,
)

env = VecFrameStack(env, n_stack=20)
eval_env = make_vec_env("neo-gathering-v0", env_kwargs=env_kwargs, n_envs=12)
_ = env.reset()


model = sb3.PPO(policy="MlpPolicy", env=env)

mean_initialiazed, std_initialized = sb3.common.evaluation.evaluate_policy(
    model, env, n_eval_episodes=N_EVAL_EPISODES, render=False
)

# eval_freq counts env.step() calls, not total timesteps.
# With N_ENVS parallel envs, divide by N_ENVS to get the right frequency.
eval_callback = EvalCallback(
    env,
    eval_freq=max(1, N_TRAIN_STEPS // (N_ENVS * 20)),
    n_eval_episodes=20,
    log_path="./logs/",
    verbose=0,
)

model.learn(total_timesteps=N_TRAIN_STEPS, progress_bar=True, callback=eval_callback)

mean_trained, std_trained = sb3.common.evaluation.evaluate_policy(
    model, env, n_eval_episodes=N_EVAL_EPISODES, render=False
)

print(f"Eval reward (init):  {mean_initialiazed:.3f} +/- {std_initialized:.3f}")
print(f"Eval reward (train): {mean_trained:.3f} +/- {std_trained:.3f}")


# # Optimal path using A*
# N_MAPS = 100
# optimal_rewards = []
# env = gym.make("neo-gathering-v0", **env_kwargs)
# for i in tqdm(range(N_MAPS), desc='Solving maps'):
#     _ = env.reset()
#     path, rewards = env.unwrapped.shortest_path()
#     optimal_rewards.append(np.sum(rewards))

# print(np.mean(optimal_rewards))


# Plot reward over training steps
timesteps = np.array(eval_callback.evaluations_timesteps)
mean_rewards = np.array([np.mean(r) for r in eval_callback.evaluations_results])
std_rewards = np.array([np.std(r) for r in eval_callback.evaluations_results])


# Figure
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(timesteps, mean_rewards, label="Mean reward")
ax.fill_between(
    timesteps,
    mean_rewards - std_rewards,
    mean_rewards + std_rewards,
    alpha=0.3,
    label="±1 std",
)
ax.axhline(
    mean_initialiazed,
    color="gray",
    linestyle="--",
    label=f"Init reward ({mean_initialiazed:.2f})",
)
ax.set_xlabel("Timesteps")
ax.set_ylabel("Reward")
ax.set_title("Reward over Training Steps (DQN)")
ax.legend()
plt.tight_layout()
# plt.savefig("training_reward.png", dpi=150)
plt.show()

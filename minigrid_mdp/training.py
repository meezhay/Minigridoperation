from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

PPO_SETTINGS = dict(n_steps=512, batch_size=64, n_epochs=10, gamma=0.99,
                    learning_rate=3e-4, ent_coef=0.01, seed=42)


def train_or_load(env_cls, path, timesteps=200_000):
    path = Path(path)
    if path.exists():
        return PPO.load(path)
    train_env = make_vec_env(env_cls, n_envs=1, wrapper_class=Monitor)
    model = PPO("MlpPolicy", train_env, verbose=0, **PPO_SETTINGS)
    model.learn(total_timesteps=timesteps)
    model.save(path)
    return model
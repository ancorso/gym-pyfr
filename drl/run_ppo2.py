import gym
import gym_pyfr
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('gym-pyfr-v0')
env.setup(['restart', '-b', 'openmp','cylinder_visc.pyfrm', 'cyl-2d-p2-start.pyfrs', 'config.ini'])
env = DummyVecEnv([lambda: env])

model = PPO2(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
model.save("pyfr-model")
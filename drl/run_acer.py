import gym
import gym_pyfr
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import ACER

env = gym.make('gym-pyfr-v0', discrete = True, n = 100)
env.setup(['restart', '-b', 'openmp','cylinder_visc.pyfrm', 'cyl-2d-p2-start.pyfrs', 'config.ini'])
env = DummyVecEnv([lambda: env])

model = ACER(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
model.save("pyfr-model")
import os
import gym
import gym_pyfr
from stable_baselines.common.policies import CnnPolicy #Change for different policies
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO #Change for different policies
from stable_baselines.bench import Monitor
from monitor_callback import get_callback

log_dir = "./trpo_log"  #Change for different policies
os.makedirs(log_dir, exist_ok=True)

env = gym.make('gym-pyfr-v0', save_dir=log_dir) # change discrete setting for different policies
env.setup(['restart', '-b', 'cuda','cylinder_visc.pyfrm', 'cyl-2d-p2-start.pyfrs', 'config.ini'])
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env]) # uncomment for other policies

model = TRPO(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=1000000, callback=get_callback(log_dir)) #Dont forget to set timesteps appropriatley
model.save(log_dir + "/pyfr-model-trpo") # change for different policies

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones: break

env.print_best()
env.print_current(log_dir + "/final_episode_performance.png")
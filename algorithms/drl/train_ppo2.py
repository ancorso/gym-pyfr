import os
import gym
import gym_pyfr
from stable_baselines.common.policies import CnnPolicy #Change for different policies
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2 #Change for different policies
from stable_baselines.bench import Monitor
from monitor_callback import get_callback

log_dir = "./ppo2_log"  #Change for different policies
os.makedirs(log_dir, exist_ok=True)

init_file = "../../init_states/coarse/Re50_shedding.pyfrs" #change
mesh_file = "../../meshes/cylinder_mesh_coarse.pyfrm"
baseline_file = "../../baseline_solutions/coarse/Re50_baseline.h5" #change
backend = "cuda" # change

env = gym.make('gym-pyfr-v0',
                mesh_file = mesh_file,
                init_file = init_file,
                baseline_file = baseline_file,
                backend = backend,
                discrete = True,
                n=50,
                save_dir=log_dir,
                Re = 50, #chnage
                tend = 500,
                plot_best_episode = True,
                )

env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env]) # uncomment for other policies

model = PPO2(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=1000000, callback=get_callback(log_dir)) #Dont forget to set timesteps appropriatley
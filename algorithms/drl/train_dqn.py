import os
import gym
import gym_pyfr
from stable_baselines.deepq.policies import CnnPolicy #Change for different policies
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN #Change for different policies
from stable_baselines.bench import Monitor
from monitor_callback import get_callback

log_dir = "./dqn_log"  #Change for different policies
os.makedirs(log_dir, exist_ok=True)

init_file = "../../init_states/coarse/Re100_shedding.pyfrs" #change
mesh_file = "../../meshes/cylinder_mesh_coarse.pyfrm"
baseline_file = "../../baseline_solutions/coarse/Re100_baseline.h5" #change
backend = "cuda" # change

env = gym.make('gym-pyfr-v0',
                mesh_file = mesh_file,
                init_file = init_file,
                baseline_file = baseline_file,
                backend = backend,
                discrete = True,
                n=50,
                save_dir=log_dir,
                Re = 100, #chnage
                tend = 500,
                dt = 0.5, #change
                plot_best_episode = True,
                action_multiplier = 0.03 #change
                )

env = Monitor(env, log_dir, allow_early_resets=True)
# env = DummyVecEnv([lambda: env]) # uncomment for other policies

model = DQN(CnnPolicy, env, verbose=1,buffer_size = 5000, prioritized_replay = True)
model.learn(total_timesteps=1000000,callback=get_callback(log_dir)) #Dont forget to set timesteps appropriatley
model.save(log_dir + "/pyfr-model-dqn")
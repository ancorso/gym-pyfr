import os
import gym
import gym_pyfr
from stable_baselines import DQN #Change for different policies

model_dir = "trained_models/dqn_cnn_model.pkl"  #Change for different policies
init_file = "../../init_states/coarse/Re50_shedding.pyfrs" #change re
mesh_file = '../../meshes/cylinder_mesh_coarse.pyfrm'
baseline_file = "../../baseline_solutions/coarse/Re50_baseline.h5" #change re

env = gym.make('gym-pyfr-v0',
                mesh_file = mesh_file,
                init_file = init_file,
                baseline_file = baseline_file,
                backend = "openmp",
                discrete = True,
                n=50,
                action_multiplier = 0.02,
                verbose = True,
                save_epsiode_animation = False,
                tend = 400) # change discrete setting for different policies

model = DQN.load(model_dir)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done: break

env.save_gif('Re' + str(env.Re) + '_suppression_anim.gif')
env.plot_current_episode('Re' + str(env.Re) + '_performance.png')
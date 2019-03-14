import os
import gym
import gym_pyfr
from stable_baselines import DQN #Change for different policies

model_dir = "dqn_cnn_model.pkl"  #Change for different policies
init_file = "../init_states/cyl-2d-p2-start.pyfrs"
mesh_file = '../meshes/cylinder_visc.pyfrm'

env = gym.make('gym-pyfr-v0', discrete = True, n=50) # change discrete setting for different policies
env.setup(['restart', '-b', 'cuda', mesh_file, init_file, 'config.ini'])

model = DQN.load(model_dir)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done: break
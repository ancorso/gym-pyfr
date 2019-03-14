import gym
import gym_pyfr

from mpi4py import MPI
MPI.init()

init_file = "../init_states/cyl-2d-p2-start.pyfrs"
mesh_file = '../meshes/cylinder_mesh_coarse.pyfrm'
env = gym.make('gym-pyfr-v0')
env.setup(['restart', '-b', 'openmp', mesh_file, init_file, 'config.ini'])

state = env.reset()
location = 88
gain = 0.4

while True:
    # Proportional controller
    rho = state[64,location,0]
    rho_v = state[64,location,2]
    omega = gain*rho_v/rho
    state, r, done, info = env.step(omega)

    if over: break


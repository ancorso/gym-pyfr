import gym
import gym_pyfr

from mpi4py import MPI
# MPI.Init()

init_file = "../../init_states/coarse/Re50_shedding.pyfrs"
mesh_file = "../../meshes/cylinder_mesh_coarse.pyfrm"
baseline_file = "../../baseline_solutions/coarse/Re50_baseline.h5"
backend = "openmp"

env = gym.make('gym-pyfr-v0',
                mesh_file = mesh_file,
                init_file = init_file,
                baseline_file = baseline_file,
                backend = backend,
                save_epsiode_animation = True,
                animation_period = 3,
                verbose = True)

state = env.reset()
location = 88
gain = .4

while True:
    # Proportional controller
    rho = state[64,location,0]
    rho_v = state[64,location,2]
    omega = gain*rho_v/rho
    state, r, done, info = env.step(omega/env.action_multiplier)

    if done: break

env.save_gif('Re' + str(env.Re) + '_suppression_anim.gif')
env.plot_current_episode('Re' + str(env.Re) + '_performance.png')

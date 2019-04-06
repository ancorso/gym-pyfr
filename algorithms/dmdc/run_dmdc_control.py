import gym
import gym_pyfr
from dmdc import DMDc
from mpc import mpc_input
from mpi4py import MPI
import numpy as np

import os

############# Common update params between runs ###############
save_dir = "DMDc_Re50_Testcase/"
init_file = "../../init_states/coarse/Re50_shedding.pyfrs"
baseline_file = "../../baseline_solutions/coarse/Re50_baseline.h5"
training_data_interval = 500
pos_during = np.concatenate([np.arange(100,200), np.arange(300, 400)])
neg_during = np.array([]) # np.concatenate([np.arange(400,500), np.arange(600,650)])


################ Setup Cylinder Environment ####################
mesh_file = "../../meshes/cylinder_mesh_coarse.pyfrm"
backend = "openmp"


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

env = gym.make('gym-pyfr-v0',
                mesh_file = mesh_file,
                init_file = init_file,
                baseline_file = baseline_file,
                backend = backend,
                save_dir = save_dir,
                verbose = True)


##################### Generate Training data ########################
print("+*+*+*+*+*+*+*+* Generating Training Data +*+*+*+*+*+*+*+*+*")
env.buffer_size  = training_data_interval
env.tend = env.buffer_size
state = env.reset()

omega_abs = 0.1
for i in range(env.buffer_size-1):
    omega = 0
    if i in pos_during:
        omega = omega_abs
    elif i in neg_during:
        omega = -omega_abs
    state, r, done, info = env.step(omega/env.action_multiplier)

# Plot the control input and "reward"
env.plot_current_episode(save_dir + "training_data.png")

# Plot the first and last state to make sure we don't have any zeros
env.plot_state(env.state_buffer[:,0].reshape(env.obs_shape), save_dir + "oldest_frame.png")
env.plot_state(env.state_buffer[:,-1].reshape(env.obs_shape), save_dir + "newest_frame.png")

################## Perfrom the DMDc ##############################
print("+*+*+*+*+*+*+* Performing DMDc +**+*+*+*+*+*+*+*+*+*+*")
Omega = np.vstack((env.state_buffer[:, :-1], np.expand_dims(env.action_buffer[:-1], axis = 0)))
Xp = env.state_buffer[:,1:]

print("Omega shape: ", Omega.shape, " Xp shape: ", Xp.shape)

A, B, P, W, transform = DMDc(Omega, Xp, retained_energy = 0.9999)

print("SHAPES -- A: ", A.shape, " B: ", B.shape, " P: ", P.shape, " W: ", W.shape, " transform: ", transform.shape)
print("retained modes: ", A.shape[0])

# Plot the B matrix for the training
if np.sum(B) == 0:
    print("B was uniformly 0, not plotting it")
else:
    B_plot = np.dot(np.linalg.pinv(transform), B).reshape((128, 256, 4))
    env.plot_state(B_plot, save_dir + "B_rho.png", dof = 0, title = "Control Response - Density")
    env.plot_state(B_plot, save_dir + "B_vx.png", dof = 1, title = "Control Response - Vx")
    env.plot_state(B_plot, save_dir + "B_vy.png", dof = 2, title = "Control Response - Vy")
    env.plot_state(B_plot, save_dir + "B_E.png", dof = 3, title = "Control Response - Energy")


############# Use DMDc + MPC for control (offline) ################
print("+*+*+*+*+* Running DMDc + MPC for offline control +**+*+*+*+*+*+*+*")
env.buffer_size = 0
env.tend = 500
env.reward_fn = lambda state : np.linalg.norm((transform @ state.flatten())[1:])
state = env.reset()
T = 32
R = 1e4,
u_max = 0.1

while True:
    x0 = transform @ state.flatten()
    omega = mpc_input(A, B, x0, T, R, u_max)
    state, r, done, info = env.step(omega/env.action_multiplier)

    if done: break

env.plot_current_episode(save_dir + 'DMDc_performance.png')

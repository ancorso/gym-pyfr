import gym
import gym_pyfr
from dmdc import DMDc
from mpc import mpc_input
from mpi4py import MPI
import numpy as np

import os

############# Common update params between runs ###############
# Training setup
reynolds_number = 50
save_dir = "DMDc_Re" + str(reynolds_number) + "_Offline_SubtractControl/"
dt = 1
training_data_interval = 700
pos_during = np.concatenate([np.arange(100,200), np.arange(300, 350)])
neg_during = np.concatenate([np.arange(400,500), np.arange(600, 650)])

# DMDc considerations
retained_energy = 0.99

# MPC params
R = 1e1
T = 16
u_max = 0.1
test_interval = 600

################ Setup Cylinder Environment ####################
mesh_file = "../../meshes/cylinder_mesh_coarse.pyfrm"
init_file = "../../init_states/coarse/Re" + str(reynolds_number) + "_shedding.pyfrs"
baseline_file = "../../baseline_solutions/coarse/Re" + str(reynolds_number) + "_baseline.h5"
backend = "openmp"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

env = gym.make('gym-pyfr-v0',
                mesh_file = mesh_file,
                init_file = init_file,
                baseline_file = baseline_file,
                backend = backend,
                save_dir = save_dir,
                dt = dt,
                Re = reynolds_number,
                verbose = True)


##################### Generate Training data #1 ########################
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
env.plot_current_episode(save_dir + "training_data1.png")

################## Perfrom the DMDc #1 ##############################
print("+*+*+*+*+*+*+* Performing DMDc # 1 Fist with all retained energy +**+*+*+*+*+*+*+*+*+*+*")
Omega = np.vstack((env.state_buffer[:, :-1], np.expand_dims(env.action_buffer[:-1], axis = 0)))
Xp = env.state_buffer[:,1:]

print("Performing DMDc with ", retained_energy, " retained energy for direct comparison")
Au, Bu, Pu, Wu, transformu = DMDc(Omega, Xp, retained_energy = retained_energy)
Bufull = np.dot(np.linalg.pinv(transformu), Bu)

print("SHAPES -- Au: ", Au.shape, " Bu: ", Bu.shape, " Pu: ", Pu.shape, " W: ", Wu.shape, " transform: ", transformu.shape)
print("retained modes from A shape: ", Au.shape[0])

# Plot the B matrix for the training
if np.sum(Bu) == 0:
    print("Bu was uniformly 0, not plotting it")
else:
    Bu_plot = Bufull.reshape((128, 256, 4))
    env.plot_state(Bu_plot, save_dir + "B_" + str(retained_energy) + "_rho.png", dof = 0, title = "Control Response - Density")
    env.plot_state(Bu_plot, save_dir + "B_" + str(retained_energy) + "_vx.png", dof = 1, title = "Control Response - Vx")
    env.plot_state(Bu_plot, save_dir + "B_" + str(retained_energy) + "_vy.png", dof = 2, title = "Control Response - Vy")
    env.plot_state(Bu_plot, save_dir + "B_" + str(retained_energy) + "_E.png", dof = 3, title = "Control Response - Energy")

##################### Generate Training data #2 ########################
print("+*+*+*+*+*+*+*+* Generating Training Data - round 2 *+*+*+*+*+*+*+*+*")
state = env.reset()

# swap the direction of the control input to get slightly different dynamics
for i in range(env.buffer_size-1):
    omega = 0
    if i in pos_during:
        omega = -omega_abs
    elif i in neg_during:
        omega = omega_abs
    state, r, done, info = env.step(omega/env.action_multiplier)

# Plot the control input and "reward"
env.plot_current_episode(save_dir + "training_data2.png")

################## Perfrom the DMDc #2 ##############################
print("+*+*+*+*+*+*+* Performing DMDc round 2 (with B subtracted off) +**+*+*+*+*+*+*+*+*+*+*")

Bsub = Bufull
Omega = np.vstack((env.state_buffer[:, :-1], np.zeros((1, env.action_buffer.shape[0]-1))))
GB = np.expand_dims(env.action_buffer[:-1], axis = 0) * Bsub
Xp = env.state_buffer[:,1:] - GB

print("Omega shape: ", Omega.shape, " Xp shape: ", Xp.shape, " GammaB shape: ", GB.shape)

As, _, Ps, Ws, transform_s = DMDc(Omega, Xp, retained_energy = retained_energy)
Bs = transform_s @ Bsub
print("SHAPES -- As: ", As.shape, " Bs: ", Bs.shape, " Ps: ", Ps.shape, " Ws: ", Ws.shape, " transform_s: ", transform_s.shape)
Bsfull = np.linalg.pinv(transform_s) @ Bs

print("SHAPES -- As: ", As.shape, " Bs: ", Bs.shape, " Ps: ", Ps.shape, " Ws: ", Ws.shape, " transform_s: ", transform_s.shape)
print("retained modes from A shape : ", As.shape[0])

# Plot the B matrix for the training
if np.sum(Bs) == 0:
    print("Bs was uniformly 0, not plotting it")
else:
    Bs_plot = Bsfull.reshape((128, 256, 4))
    env.plot_state(Bs_plot, save_dir + "B_sub_rho.png", dof = 0, title = "Control Response - Density")
    env.plot_state(Bs_plot, save_dir + "B_sub_vx.png", dof = 1, title = "Control Response - Vx")
    env.plot_state(Bs_plot, save_dir + "B_sub_vy.png", dof = 2, title = "Control Response - Vy")
    env.plot_state(Bs_plot, save_dir + "B_sub_E.png", dof = 3, title = "Control Response - Energy")


############# Use DMDc + MPC for control (offline) ################
print("+*+*+*+*+* Running DMDc + MPC for offline control +**+*+*+*+*+*+*+*")
env.buffer_size = 0
env.tend = test_interval
env.save_epsiode_animation = True
env.reward_fn = lambda state : -np.linalg.norm((transform_s @ state.flatten())[1:])
state = env.reset()


while True:
    x0 = transform_s @ state.flatten()
    omega = mpc_input(As, Bs, x0, T, R, u_max)
    state, r, done, info = env.step(omega/env.action_multiplier)

    if done: break

env.save_gif(save_dir + 'suppression_anim.gif')
env.plot_current_episode(save_dir + 'DMDc_performance.png')

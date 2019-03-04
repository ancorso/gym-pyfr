import matplotlib.pyplot as plt

import gym
import gym_pyfr
env = gym.make('gym-pyfr-v0')
env.setup(['restart', '-b', 'cuda','cylinder_visc.pyfrm', 'cyl-2d-p2-start.pyfrs', 'config.ini'])

# plt.figure()
# CS = plt.contourf(env.pyfr.goal_state.reshape(128,256,4)[:,:,2]) #, cc, zz_miss)
# # CS = plt.contourf(state[:,:,2]) #, cc, zz_miss)
#
# nm, lbl = CS.legend_elements()
# plt.legend(nm, lbl, title= 'MyTitle', fontsize= 8)
#
# plt.show()
state = env.pyfr.get_state()
location = 88
gain = 0.4
reward = []
control = []
time = []
print("starting iteration")
while True:
    rho = state[64,location,0]
    rho_v = state[64,location,2]
    omega = gain*rho_v/rho
    state, r, over, info = env.step(omega)

    reward.append(r)
    control.append(omega)
    time.append(info["timestep"])
    print("omega: ", omega, "reward: ", r, "iteration: ", time[-1])

    if over:
        break

print("sum of rewards: ", sum(reward))

env.finalize()

plt.figure()
plt.subplot(1,2,1)
plt.plot(time, reward)
plt.subplot(1,2,2)
plt.plot(time, control)
plt.savefig("suppression_performance")
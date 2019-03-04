import matplotlib.pyplot as plt

import gym
import gym_pyfr
from baselines import deepq


env = gym.make('gym-pyfr-v0', discrete = True)
env.setup(['restart', '-b', 'cuda','cylinder_visc.pyfrm', 'cyl-2d-p2-start.pyfrs', 'config.ini'])



act = deepq.learn(
    env,
    network='mlp',
    lr=1e-3,
    total_timesteps=50000,
    buffer_size=500,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    print_freq=10,
    target_network_update_freq=100
)
print("Saving model to cartpole_model.pkl")
act.save("cartpole_model.pkl")

env.finalize()

plt.figure()
plt.subplot(1,2,1)
plt.plot(time, reward)
plt.subplot(1,2,2)
plt.plot(time, control)
plt.savefig("suppression_performance")
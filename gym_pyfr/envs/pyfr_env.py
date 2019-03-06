import gym
import pyfr
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_pyfr.envs.pyfr_obj import PyFRObj
from collections import deque
from copy import copy
import matplotlib.pyplot as plt

def print_trace(rewards, actions, episode_str, filename):
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.plot(range(len(rewards)), rewards)
    plt.title('Episode ' + episode_str + ' Reward')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')

    plt.subplot(1,2,2)
    plt.plot(range(len(actions)), actions)
    plt.title('Episode ' + episode_str + ' Action')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')

    plt.savefig(filename)


class PyFREnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                discrete = False,
                n = 20,
                action_multiplier = 0.01,
                verbose = False,
                save_dir = ".",
                print_on_iteration = 100
                ):

        # Keep track of logging information
        self.verbose = verbose
        self.save_dir = save_dir
        self.episode = -1
        self.best_reward = -float('inf')
        self.best_reward_sequence = []
        self.best_action_sequence = []
        self.best_episode = -1
        self.print_on_iteration = print_on_iteration

        # Setup omega range
        self.action_multiplier = action_multiplier
        self.omega_min = -2*action_multiplier
        self.omega_max = 2*action_multiplier
        self.d_omega = (self.omega_max - self.omega_min)/n

        # Setup the observation and action spaces
        self.discrete = discrete
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(128, 256, 4), dtype=np.float64)
        if discrete:
            print("Initializing discrete action space with n=",n)
            self.action_space = spaces.Discrete(n)
        else:
            print("Initializing continuous action space with action_multiplier=",self.action_multiplier)
            self.action_space = spaces.Box(low=-2, high=2, shape=(1,), dtype=np.float64)

        # Build the pyf object run run things on
        self.pyfr = PyFRObj()

    def setup(self, cmd_args):
        self.episode += 1
        print('parsing with cmd args: ', cmd_args)
        self._cmd_args = cmd_args
        self.pyfr.parse(cmd_args)
        self.pyfr.process()
        self.pyfr.setup_dataframe()
        self.pyfr.solver.tlist = deque(range(int(self.pyfr.solver.tcurr), int(self.pyfr.solver.tlist[-1])))
        self.iteration = 0
        self.current_reward_sequence = []
        self.current_action_sequence = []


    def step(self, action):
        # Set action
        if self.discrete:
            action = self.omega_min + action*self.d_omega
        else:
            action = self.action_multiplier*action

        self.pyfr.take_action(action)

        # Step the simulation to the next timestep
        episode_over = self.pyfr.step()

        # Get the new state
        ob = self.pyfr.get_state()

        # Get the reward
        reward = self.pyfr.get_reward(ob)

        # No info yet
        info = {"timestep":self.pyfr.solver.tcurr}

        # Print step information
        if episode_over or self.verbose or self.iteration % self.print_on_iteration == 0:
            print("Episode: ", self.episode, " Iteration: ", self.iteration, " Action: ", action, " Reward: ", reward)

        # update sequences and iterations
        self.iteration += 1
        self.current_action_sequence.append(action)
        self.current_reward_sequence.append(reward)

        # Handle end of episode business
        if episode_over:
            self.end_of_episode()

        # Return the results of the step
        return ob, reward, episode_over, info

    def end_of_episode(self):
        total_reward = sum(self.current_reward_sequence)
        print("Episode over, total reward: ", total_reward)
        if total_reward > self.best_reward:
            print("Found new best reward! Overwriting old traces")
            self.best_reward = total_reward
            self.best_episode = self.episode
            self.best_action_sequence = copy(self.current_action_sequence)
            self.best_reward_sequence = copy(self.current_reward_sequence)
            self.print_best()


    def print_best(self):
        print("Printing Best... The best episode was ", self.best_episode, " out of ", self.episode)
        fname = self.save_dir + "/performance_best_episode_"+str(self.episode)+".png"
        print_trace(self.best_reward_sequence, self.best_action_sequence, str(self.episode) + " (current best)", fname)

    def print_current(self, fname = None):
        if fname is None:
            fname = self.save_dir + "/performance_episode_"+str(self.episode) + ".png"
        print_trace(self.current_reward_sequence, self.current_action_sequence, str(self.episode), fname)


    # Return the state
    def reset(self):
        if self.verbose:
            print("Resetting...\n")
        self.setup(self._cmd_args)
        return self.pyfr.get_state()

    def finalize(self):
        self.pyfr.finalize()




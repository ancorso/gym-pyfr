import gym
import pyfr
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_pyfr.envs.pyfr_obj import PyFRObj
from gym_pyfr.envs.plot_utils import plot_rewards_and_actions, plot_state, make_gif
from collections import deque
from copy import copy

class PyFREnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                mesh_file, # The location of the mesh used by PyFR
                init_file, # The initial solution file *.pyfrs that PyFR uses to initialize
                config_file, # The PyFR configuration file
                baseline_file = None, # The baseline solution file to compare the state to (to compute reward)
                backend = "cuda", # The PyFR backend
                discrete = False, # Whether or not to discretize the action space
                n = 20, # The number of actions to discretize the action space to
                action_multiplier = 0.01, # Multiplier on the actions (the space is set from -2 to 2 so that initially there is not cutoff in the network)
                verbose = False, # Whether or not to display more information
                save_dir = ".", # The directory to save plots and
                print_on_iteration = 100, # Frequency of printing stats when verbose is off
                plot_best_episode = True, # Whether or not to plot the reward and action vs iteration and any new best rewards
                save_epsiode_animation = False, # Whether or not to create an animation of each episode
                animation_period = 1 # timesteps between animation frames
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
        self.plot_best_episode_flag = plot_best_episode
        self.save_epsiode_animation = save_epsiode_animation
        self.animation_period = animation_period

        # Setup omega range
        self.action_multiplier = action_multiplier
        self.omega_min = -2*action_multiplier
        self.omega_max = 2*action_multiplier
        self.d_omega = (self.omega_max - self.omega_min)/n

        # Setup the observation and action spaces
        self.discrete = discrete
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(128, 256, 4), dtype=np.float64)
        if discrete:
            if self.verbose: print("Initializing discrete action space with n=",n)
            self.action_space = spaces.Discrete(n)
        else:
            if self.verbose: print("Initializing continuous action space with action_multiplier=",self.action_multiplier)
            self.action_space = spaces.Box(low=-2, high=2, shape=(1,), dtype=np.float64)

        # Build the pyf object run run things on
        self.pyfr = PyFRObj()

        # Setup the pyfr object
        self.mesh_file = mesh_file
        self.init_file = init_file
        self.config_file = config_file
        self.baseline_file = baseline_file
        self.backend = backend
        self.setup()

    def setup(self):
        self.episode += 1
        cmd_args = ['restart', '-b', self.backend, self.mesh_file, self.init_file, self.config_file]
        if self.verbose: print('parsing with cmd args: ', cmd_args)
        self.pyfr.parse(cmd_args)
        self.pyfr.set_baseline(self.baseline_file)
        self.pyfr.process()
        self.pyfr.setup_dataframe()
        self.pyfr.solver.tlist = deque(range(int(self.pyfr.solver.tcurr), int(self.pyfr.solver.tlist[-1]) + 1))
        self.iteration = 0
        self.current_reward_sequence = []
        self.current_action_sequence = []
        self.animation = []


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

        # Plot if necessary
        if self.save_epsiode_animation and self.iteration % self.animation_period == 0:
            self.animation.append(plot_state(ob, 2, "Y-Velocity at Iteration " + str(self.iteration)))

        # update sequences and iterations
        self.iteration += 1
        self.current_action_sequence.append(action)
        self.current_reward_sequence.append(reward)

        # Handle end of episode business
        if episode_over:
            self.end_of_episode()

        # Return the results of the step
        return ob, reward, episode_over, info

    # Take care of anything that happens at the end of an episode (before reset)
    def end_of_episode(self):
        if self.save_epsiode_animation:
            make_gif(self.animation, "Episode_" + str(self.episode) + "_anim.gif")

        total_reward = sum(self.current_reward_sequence)
        print("Episode over, total reward: ", total_reward)
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_episode = self.episode
            self.best_action_sequence = copy(self.current_action_sequence)
            self.best_reward_sequence = copy(self.current_reward_sequence)
            if self.plot_best_episode_flag:
                self.plot_best_episode()


    def plot_best_episode(self):
        print("Plotting best episode (", self.best_episode, " out of ", self.episode, ")")
        fname = self.save_dir + "/performance_best_episode_"+str(self.episode)+".png"
        plot_rewards_and_actions(self.best_reward_sequence, self.best_action_sequence, str(self.episode) + " (current best)", fname)

    def plot_latest_episode(self, fname = None):
        if fname is None:
            fname = self.save_dir + "/performance_episode_"+str(self.episode) + ".png"
        plot_rewards_and_actions(self.current_reward_sequence, self.current_action_sequence, str(self.episode), fname)


    # Return the state
    def reset(self):
        if self.verbose: print("Resetting...")
        self.setup()
        return self.pyfr.get_state()

    # init mpi
    def init_mpi(self):
        self.pyfr.init()

    # Finalize mpi
    def finalize_mpi(self):
        self.pyfr.finalize()




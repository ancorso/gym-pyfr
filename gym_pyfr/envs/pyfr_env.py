import gym
import pyfr
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_pyfr.envs.pyfr_obj import PyFRObj
from gym_pyfr.envs.plot_utils import plot_rewards_and_actions, plot_state, make_gif
from collections import deque
from copy import copy
import os
import math
import h5py

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

class PyFREnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                mesh_file, # The location of the mesh used by PyFR
                init_file = None, # The initial solution file *.pyfrs that PyFR uses to initialize
                config_file = os.path.join(__location__, 'config_base.ini'), # The PyFR configuration file
                baseline_file = None, # The baseline solution file to compare the state to (to compute reward)
                backend = "cuda", # The PyFR backend
                discrete = False, # Whether or not to discretize the action space
                n = 20, # The number of actions to discretize the action space to
                action_multiplier = 0.01, # Multiplier on the actions (the space is set from -2 to 2 so that initially there is not cutoff in the network)
                verbose = False, # Whether or not to display more information
                save_dir = ".", # The directory to save plots and models
                sol_dir = 'sol_data', # directory to store solution data in
                print_period = 100, # Frequency of printing stats when verbose is off
                plot_best_episode = False, # Whether or not to plot the reward and action vs iteration and any new best rewards
                save_epsiode_animation = False, # Whether or not to create an animation of each episode
                animation_period = 1, # timesteps between animation frames
                Re = None, # reynolds number override
                tend = None, # end time override
                write_state_files = False, # Whether or not to save the state files
                write_state_period = 1 # Period of saving state files
                ):

        # Keep track of logging information
        self.verbose = verbose
        self.save_dir = save_dir
        self.sol_dir = sol_dir
        self.episode = -1
        self.best_reward = -float('inf')
        self.best_reward_sequence = []
        self.best_action_sequence = []
        self.best_episode = -1
        self.print_period = print_period
        self.plot_best_episode_flag = plot_best_episode
        self.save_epsiode_animation = save_epsiode_animation
        self.animation_period = animation_period
        self.write_state_period = write_state_period
        self.write_state_files = write_state_files

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
        self.Re = Re
        self.tend = tend
        self.setup()


    def setup(self):
        self.episode += 1 # increment the episode count for this instance

        # Load the command line arguments
        if self.init_file is None:
            cmd_args = ['run', '-b', self.backend, self.mesh_file, self.config_file]
        else:
            cmd_args = ['restart', '-b', self.backend, self.mesh_file, self.init_file, self.config_file]
        if self.verbose: print('parsing with cmd args: ', cmd_args)
        self.pyfr.parse(cmd_args) # this loads in the config file but does not prcess it yet

        #update any changes to the config file we would like to make
        self.pyfr.baseline_file = self.baseline_file
        if self.Re is not None:
            gamma = self.pyfr.cfg.getfloat('constants', 'gamma')
            M = self.pyfr.cfg.getfloat('constants', 'M')
            u = math.sqrt(gamma)*M
            mu = u/self.Re
            if self.verbose: print('setting mu =', str(mu), ' to get Re =', self.Re)
            self.pyfr.cfg.set('constants', 'mu', str(mu))
        if self.tend is not None:
            if self.verbose: print('Setting tend =', self.tend)
            self.pyfr.cfg.set('solver-time-integrator', 'tend', self.tend)

        # Setup the rest of the pyfr object
        self.pyfr.process()
        self.pyfr.setup_dataframe()
        self.pyfr.solver.tlist = deque(range(int(self.pyfr.solver.tcurr), int(self.pyfr.solver.tlist[-1]) + 1))

        # reset per-epsiode parameters
        self.iteration = 0
        self.current_reward_sequence = []
        self.current_action_sequence = []
        self.animation = []
        self.curr_state = self.pyfr.get_state()
        self.curr_reward = self.pyfr.get_reward(self.curr_state)


    def save_native(self, save_dir, basename, t = 0):
        self.pyfr.save_solution(save_dir, basename, t)


    def save_h5(self, basename = 'sol_data_', save_dir = None, t = 0):
        if save_dir is None:
            save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        filename = save_dir + '/' + basename + str(int(t)).zfill(4) + '.h5'
        f = h5py.File(filename, 'w')
        f['sol_data'] = self.curr_state
        f['control_input'] = self.pyfr.last_action
        f['reward'] = self.curr_reward
        f.close()

    # Run the simulation until it stops
    def run(self):
        while True:
            state, r, done, info = self.step(0)
            if done: break


    # Step the simulation forward in time after applying the action.
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
        self.curr_state = self.pyfr.get_state()

        # Get the reward
        self.curr_reward = self.pyfr.get_reward(self.curr_state)

        # No info yet
        info = {"timestep": self.pyfr.solver.tcurr}

        # Print step information
        if episode_over or self.verbose or self.iteration % self.print_period == 0:
            print("Episode: ", self.episode, " Iteration: ", self.iteration, " Action: ", action, " Reward: ", self.curr_reward)

        # Plot if necessary
        if self.save_epsiode_animation and self.iteration % self.animation_period == 0:
            self.animation.append(plot_state(self.curr_state, 2, "Y-Velocity at Iteration " + str(self.iteration)))

        # Save output if necessary
        if self.write_state_files and self.iteration % self.write_state_period == 0:
            self.save_h5(save_dir = self.save_dir + '/' + self.sol_dir, t = self.pyfr.solver.tcurr)

        # update sequences and iterations
        self.iteration += 1
        self.current_action_sequence.append(action)
        self.current_reward_sequence.append(self.curr_reward)

        # Handle end of episode business
        if episode_over:
            self.end_of_episode()

        # Return the results of the step
        return self.curr_state, self.curr_reward, episode_over, info


    def save_gif(self, filename = None):
        if filename is None:
            filename = "Episode_" + str(self.episode) + "_anim.gif"
        make_gif(self.animation, filename)

    # Take care of anything that happens at the end of an episode (before reset)
    def end_of_episode(self):
        total_reward = sum(self.current_reward_sequence)
        print("Episode over, total reward: ", total_reward)
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_episode = self.episode
            self.best_action_sequence = copy(self.current_action_sequence)
            self.best_reward_sequence = copy(self.current_reward_sequence)
            if self.plot_best_episode_flag:
                self.plot_best_episode()


    # Plot the rewards and actions as a function of iteration for the best episode seen so far
    def plot_best_episode(self):
        print("Plotting best episode (", self.best_episode, " out of ", self.episode, ")")
        fname = self.save_dir + "/performance_best_episode_"+str(self.episode)+".png"
        plot_rewards_and_actions(self.best_reward_sequence, self.best_action_sequence, str(self.episode) + " (current best)", fname)


    # Plot the rewards and actions as a function of iteration for the current episode
    def plot_current_episode(self, fname = None):
        if fname is None:
            fname = self.save_dir + "/performance_episode_"+str(self.episode) + ".png"
        plot_rewards_and_actions(self.current_reward_sequence, self.current_action_sequence, str(self.episode), fname)


    # Plot the current state of the system
    def plot_state(self, fname, dof = 2, title = 'Y-Velocity'):
        plot_state(self.curr_state, dof, title, outfile = fname)


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




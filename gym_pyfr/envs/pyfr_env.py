import gym
import pyfr
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_pyfr.envs.pyfr_obj import PyFRObj
from collections import deque

class PyFREnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, discrete = False, n = 20, omega_min = -0.1, omega_max = 0.1):
        print("initiating")

        self.discrete = discrete
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.d_omega = (self.omega_max - self.omega_min)/n

        # Setup the observation and action spaces
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(128, 256, 4), dtype=np.float64)
        if discrete:
            self.action_space = spaces.Discrete(n)
        else:
            self.action_space = spaces.Box(low=omega_min, high=omega_max, shape=(1,), dtype=np.float64)

        # Build the pyf object run run things on
        self.pyfr = PyFRObj()

    def setup(self, cmd_args):
        print('parsing with cmd args: ', cmd_args)
        self._cmd_args = cmd_args
        self.pyfr.parse(cmd_args)
        self.pyfr.process()
        self.pyfr.setup_dataframe()
        self.pyfr.solver.tlist = deque(range(int(self.pyfr.solver.tcurr), int(self.pyfr.solver.tlist[-1])))

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        # Set action
        if self.discrete:
            action = self.omega_min + action*self.d_omega

        self.pyfr.take_action(action)

        # Step the simulation to the next timestep
        episode_over = self.pyfr.step()

        # Get the new state
        ob = self.pyfr.get_state()

        # Get the reward
        reward = self.pyfr.get_reward(ob)
        print("step called with: ", action, " got reward: ", r)

        # No info yet
        info = {"timestep":self.pyfr.solver.tcurr}

        return ob, reward, episode_over, info

    # Return the state
    def reset(self):
        self.setup(self._cmd_args)
        return self.pyfr.get_state()

    def finalize(self):
        self.pyfr.finalize()




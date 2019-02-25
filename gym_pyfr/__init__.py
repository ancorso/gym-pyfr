from gym.envs.registration import register

register(
    id='gym-pyfr-v0',
    entry_point='gym_pyfr.envs:PyFREnv',
)
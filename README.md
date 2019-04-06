# gym-pfyr
Gym environment that runs PyFR on a 2D flow over a circular cylinder with rotational control.

![](suppression_example.gif)


### Installation
1. Install `gym` from [here](https://gym.openai.com/docs/)
2. Install `PyFR` version [v1.7.6](https://github.com/vincentlab/PyFR/releases/tag/v1.7.6)
3. Copy `gym.patch` to the PyFR directory and use `git apply gym.patch` to apply it
4. From the top level directory, `gym-pyfr`, run `pip install -e .`

### Contents
* `algorithms/` - Contains sample algorithms for solving the vortex suppression problem. Includes Deep RL and proportional controller approaches
* `baseline_solutions/` - Contains steady-state solutions for different Reynolds number flows around a circular cylinder. Note that these are approximate
* `gym_pyfr/` - This folder contains the `gym` environment code
* `init_states/` - Contains solution states where vortex shedding has already started for different Reynolds numbers
* `meshes/` - Contains the available meshes for the 2D circular cylinder

### PyFR environment Constructor Arguments
The following are the options available when constructing a gym-pyfr environment. Default values are given with `argument = default_val`.

* `mesh_file` - The location of the mesh used by PyFR
* `init_file = None` - The initial solution file *.pyfrs that PyFR uses to initialize
* `config_file = os.path.join(__location__, 'config_base.ini')` - The PyFR configuration file
* `baseline_file = None` - The baseline solution file to compare the state to (to compute reward)
* `backend = "cuda"` - The PyFR backend
* `discrete = False` - Whether or not to discretize the action space
* `n = 20` - The number of actions to discretize the action space to
* `action_multiplier = 0.01` -  Multiplier on the actions (the space is set from -2 to 2 so that initially there is not cutoff in the network)
* `verbose = False` - Whether or not to display more information
* `save_dir = "."` - The directory to save plots and models
* `sol_dir = 'sol_data'` - directory to store solution data in
* `print_period = 100` -  Frequency of printing stats when verbose is off
* `plot_best_episode = False` - Whether or not to plot the reward and action vs iteration and any new best rewards
* `save_epsiode_animation = False` - Whether or not to create an animation of each episode
* `animation_period = 1` - Number of timesteps between animation frames
* `Re = None` - Reynolds number override
* `tend = None` - end time override
* `write_state_files = False` -  Whether or not to save the state files
* `write_state_period = 1` -  Period of saving state files


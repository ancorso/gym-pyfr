# gym-pfyr
Gym environment that runs PyFR

### PyFR environment Constructor Arguments
* mesh_file, # The location of the mesh used by PyFR
* init_file = None, # The initial solution file *.pyfrs that PyFR uses to initialize
* config_file = os.path.join(__location__, 'config_base.ini'), # The PyFR configuration file
* baseline_file = None, # The baseline solution file to compare the state to (to compute reward)
* backend = "cuda", # The PyFR backend
* discrete = False, # Whether or not to discretize the action space
* n = 20, # The number of actions to discretize the action space to
* action_multiplier = 0.01, # Multiplier on the actions (the space is set from -2 to 2 so that initially there is not cutoff in the network)
* verbose = False, # Whether or not to display more information
* save_dir = ".", # The directory to save plots and models
* sol_dir = 'sol_data', # directory to store solution data in
* print_period = 100, # Frequency of printing stats when verbose is off
* plot_best_episode = False, # Whether or not to plot the reward and action vs iteration and any new best rewards
* save_epsiode_animation = False, # Whether or not to create an animation of each episode
* animation_period = 1, # timesteps between animation frames
* Re = None, # reynolds number override
* tend = None, # end time override
* write_state_files = False, # Whether or not to save the state files
* write_state_period = 1 # Period of saving state files
* 
### Directory Descriptions

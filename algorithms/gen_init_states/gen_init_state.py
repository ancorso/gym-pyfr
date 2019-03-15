import gym
import gym_pyfr
import os

from mpi4py import MPI
# MPI.Init()

desired_Re_list = [50, 75, 100, 125, 150, 200]
mesh_file = "../../meshes/cylinder_mesh_coarse.pyfrm"
backend = "cuda"

startup_Re = 100
filename = 'cyl-2d-p2-000.pyfrs'
for desired_Re in desired_Re_list:
    print('Generating shedding for Re = ', startup_Re, '...')
    env = gym.make('gym-pyfr-v0',
                    mesh_file = mesh_file,
                    baseline_file = "../../baseline_solutions/re50_base.h5",
                    backend = backend,
                    save_epsiode_animation = True,
                    animation_period = 3,
                    Re = max(desired_Re, startup_Re),
                    tend = 4,
                    verbose = True
                    )

    # Run the coarse mesh for 400 timesteps
    print('Running rampup to Re =', env.Re)
    env.run()
    basename = 'Re' + str(env.Re) + '_rampup'
    env.save_gif(basename + '.gif')
    env.plot_current_episode(basename + '.png')

    # If we now have to lower the reynolds number, do so
    if desired_Re < startup_Re:
        print('Starting cooldown at Re = ', env.Re)

        # Save the solution file in order to restart
        env.save_native('.', 'cyl-2d-p2-{n:03d}', t = 0)

        # update parameters to get the desired reynolds number shedding
        env.init_file = filename
        env.Re = desired_Re
        env.tend = 6
        env.reset()
        env.run()
        basename = 'Re' + str(env.Re) + '_cooldown'
        env.save_gif(basename + '.gif')
        env.plot_current_episode(basename + '.png')
        os.remove(filename)

    # We have shedding at the desired Re so save the results
    print('Saving init file to: ', outputname, '...')
    env.plot_state('Re'+str(desired_Re)+'_init_y_vel.png')
    env.save_native('.', 'cyl-2d-p2-{t:03d}', t = 0)
    outputname = 'Re'+str(desired_Re)+'_shedding.pyfrs'
    os.rename(filename, outputname)



    # Get 100 solution steps to use for dmdc-produced baseline
    print('Generating extra data for DMDc basline...')
    env.init_file = outputname
    env.tend = 2
    env.write_state_files = True
    env.save_epsiode_animation = False
    env.sol_dir = 'sol_data_Re' + str(env.Re)
    env.reset()
    env.run()
    env.plot_current_episode('Re' + str(env.Re) + '_datagen' + '.png')






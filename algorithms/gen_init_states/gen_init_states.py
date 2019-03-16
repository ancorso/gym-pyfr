import gym
import gym_pyfr
import os

from mpi4py import MPI
# MPI.Init()

desired_Re_list = [50, 75, 100, 125, 150, 200]
mesh_file = "../../meshes/cylinder_mesh_coarse.pyfrm"
backend = "openmp"

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
                    tend = 400,
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
        env.tend = 800
        env.reset()
        env.run()
        basename = 'Re' + str(env.Re) + '_cooldown'
        env.save_gif(basename + '.gif')
        env.plot_current_episode(basename + '.png')
        os.remove(filename)

    # We have shedding at the desired Re so save the results
    env.plot_curr_state('Re'+str(desired_Re)+'_init_y_vel.png')
    env.save_native('.', 'cyl-2d-p2-{t:03d}', t = 0)
    outputname = 'Re'+str(desired_Re)+'_shedding.pyfrs'
    print('Saving init file to: ', outputname, '...')
    os.rename(filename, outputname)



    # Get 100 solution steps to use for dmdc-produced baseline
    print('Generating extra data for DMDc basline...')
    env.init_file = outputname
    env.tend = 210
    env.save_epsiode_animation = False
    avg_state = env.reset()
    divisor = 1
    while True:
        state, r, done, info = env.step(0)
        avg_state += state
        divisor += 1
        if done: break
    baseline = avg_state / divisor

    env.save_h5(baseline, env.save_dir, 'Re' + str(env.Re) + '_baseline.h5')
    env.plot_state(baseline, 'Re' + str(env.Re) + '_baseline.png', dof = 2, title = 'Y-Velocity')
    env.plot_current_episode('Re' + str(env.Re) + '_datagen.png')






import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
import imageio

def plot_rewards_and_actions(rewards, actions, episode_str, filename):
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

def plot_state(state, dof, title, outfile = None):
    fig = plt.figure(0)
    fig.clf()
    plt.title(title)
    X, Y = np.meshgrid(range(1, 257), range(1, 129))
    Z = gaussian_filter(state[:,:,dof], sigma=2)
    plt.contour(X,Y,Z, levels=20)
    plt.colorbar()

    if outfile is not None:
        plt.savefig(outfile)

    fig.canvas.draw()       # draw the canvas, cache the renderer
    w, h = fig.canvas.get_width_height()
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(2*h, 2*w, 3)
    return image

def make_gif(image_list, outfile):
    imageio.mimsave(outfile, image_list, fps=20)


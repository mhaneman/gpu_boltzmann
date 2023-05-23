import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Plotting:
    def __init__(self) -> None:
        pass


    def animate_simulation(self, radius, ixr, ixl, rs, vs):
        # expectation curve
        v = np.linspace(0, 2000, 1000)
        a = 2/500**2
        fv = a*v*np.exp(-a*v**2 / 2)

        # plot simulation
        bins = np.linspace(0,1500,50)
        fig, axes = plt.subplots(1, 2, figsize=(20,10))
        axes[0].clear()
        vmin = 0
        vmax = 1
        axes[0].set_xlim(0,1)
        axes[0].set_ylim(0,1)
        markersize = 2 * radius * axes[0].get_window_extent().width  / (vmax-vmin) * 72./fig.dpi
        red, = axes[0].plot([], [], 'o', color='red', markersize=markersize)
        blue, = axes[0].plot([], [], 'o', color='blue', markersize=markersize)
        n, bins, patches = axes[1].hist(torch.sqrt(torch.sum(vs[0]**2, axis=0)).cpu(), bins=bins, density=True)
        axes[1].plot(v,fv)
        axes[1].set_ylim(top=0.003)

        def animate(i):
            xred, yred = rs[i][0][ixr].cpu(), rs[i][1][ixr].cpu()
            xblue, yblue = rs[i][0][ixl].cpu(),rs[i][1][ixl].cpu()
            red.set_data(xred, yred)
            blue.set_data(xblue, yblue)
            hist, _ = np.histogram(torch.sqrt(torch.sum(vs[i]**2, axis=0)).cpu(), bins=bins, density=True)
            for i, patch in enumerate(patches):
                patch.set_height(hist[i])
            return red, blue

        writer = animation.FFMpegWriter(fps=30)
        ani = animation.FuncAnimation(fig, animate, frames=500, interval=50, blit=True)
        ani.save('my_sim_gpu.mp4',writer=writer,dpi=100)
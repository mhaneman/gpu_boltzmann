import torch

from gpu_particles import GpuParticles
from plotting import Plotting


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_particles = GpuParticles(device=device, num_of_particles=10_000, radius=0.001)
    rs, vs = gpu_particles.motion(ts=1000, dt=0.000008)

    plot = Plotting()
    plot.animate_simulation(gpu_particles.radius, gpu_particles.particles_right, gpu_particles.particles_left, rs, vs)
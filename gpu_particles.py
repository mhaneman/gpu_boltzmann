import torch

class GpuParticles:
    def __init__(self, device, num_of_particles, radius) -> None:
        self.device = device
        self.num_of_particles = num_of_particles
        self.radius = radius
        self.d_cutoff=2*radius
        
        # initalize position and velocities
        self.positions = torch.rand((2, self.num_of_particles)).to(device)
        self.particles_left = self.positions[0] < 0.5
        self.particles_right = self.positions[0] > 0.5

        self.velocities = torch.zeros((2, self.num_of_particles)).to(device)
        self.velocities[0][self.particles_left] = 500
        self.velocities[0][self.particles_right] = -500

        # relationship between particles
        self.ids = torch.arange(self.num_of_particles)
        self.ids_pairs = torch.combinations(self.ids, 2).to(device)


    def get_delta_d2_pairs(self, r, ids_pairs):
        dx = torch.diff(torch.stack([r[0][ids_pairs[:,0]], r[0][ids_pairs[:,1]]]).T).squeeze()
        dy = torch.diff(torch.stack([r[1][ids_pairs[:,0]], r[1][ids_pairs[:,1]]]).T).squeeze()
        return dx**2 + dy**2


    def compute_new_v(self, v1, v2, r1, r2):
        v1new = v1 - torch.sum((v1-v2)*(r1-r2), axis=0) / torch.sum((r1-r2)**2, axis=0) * (r1-r2)
        v2new = v2 - torch.sum((v1-v2)*(r1-r2), axis=0) / torch.sum((r2-r1)**2, axis=0) * (r2-r1)
        return v1new, v2new


    # update motion
    def motion(self, ts, dt):
        rs = torch.zeros((ts, self.positions.shape[0], self.positions.shape[1])).to(self.device)
        vs = torch.zeros((ts, self.velocities.shape[0], self.velocities.shape[1])).to(self.device)
        # Initial State
        rs[0] = self.positions
        vs[0] = self.velocities

        for i in range(1,ts):
            ic = self.ids_pairs[self.get_delta_d2_pairs(self.positions, self.ids_pairs) < self.d_cutoff**2] # get all particle pairs that are colliding
            self.velocities[:,ic[:,0]], self.velocities[:,ic[:,1]] = self.compute_new_v(self.velocities[:,ic[:,0]], self.velocities[:,ic[:,1]], self.positions[:,ic[:,0]], self.positions[:,ic[:,1]]) # update velocities
            
            # bounce off walls
            self.velocities[0,self.positions[0]>1] = -torch.abs(self.velocities[0,self.positions[0]>1])
            self.velocities[0,self.positions[0]<0] = torch.abs(self.velocities[0,self.positions[0]<0])
            self.velocities[1,self.positions[1]>1] = -torch.abs(self.velocities[1,self.positions[1]>1])
            self.velocities[1,self.positions[1]<0] = torch.abs(self.velocities[1,self.positions[1]<0])
            
            # update position
            self.positions = self.positions + self.velocities*dt

            #update tensor
            rs[i] = self.positions
            vs[i] = self.velocities

        return rs, vs

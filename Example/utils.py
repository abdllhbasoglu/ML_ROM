import torch
import numpy as np
import os
torch.set_default_dtype(torch.float64)


class Reconstructed_writer:
    def __init__(self, dataset, phat_file, uhat_file, vhat_file, num_cases, num_rows, num_cols, num_channels):
        self.dataset = dataset
        self.phat_file = phat_file
        self.uhat_file = uhat_file
        self.vhat_file = vhat_file
        self.num_cases = num_cases
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_channels = num_channels


    def write_dataset(self):
        for i in range(self.num_cases):

            # File name "1_output_p.dat", "1_output_u.dat", "1_output_v.dat" and go on
            pressure_file = os.path.join(self.phat_file, f"{i+1}_output_p.dat")
            u_velocity_file = os.path.join(self.uhat_file, f"{i+1}_output_u.dat")
            v_velocity_file = os.path.join(self.vhat_file, f"{i+1}_output_v.dat")

            #CPU
            #pressure_data = self.dataset[i,0,:,:].detach().numpy().reshape(self.num_rows*self.num_cols, 1)
            #u_velocity_data = self.dataset[i,1,:,:].detach().numpy().reshape(self.num_rows*self.num_cols, 1)
            #v_velocity_data = self.dataset[i,2,:,:].detach().numpy().reshape(self.num_rows*self.num_cols, 1)

            #CUDA
            pressure_data = self.dataset[i,0,:,:].cpu().detach().numpy().reshape(self.num_rows*self.num_cols, 1)
            u_velocity_data = self.dataset[i,1,:,:].cpu().detach().numpy().reshape(self.num_rows*self.num_cols, 1)
            v_velocity_data = self.dataset[i,2,:,:].cpu().detach().numpy().reshape(self.num_rows*self.num_cols, 1)

            np.savetxt(pressure_file, pressure_data, fmt='%.9e')
            np.savetxt(u_velocity_file, u_velocity_data, fmt='%.9e')
            np.savetxt(v_velocity_file, v_velocity_data, fmt='%.9e')

import numpy as np
import torch
import torchvision
from numpy import sqrt
from torch import nn

import ising
from args import args
from bernoulli import BernoulliMixture
from pixelcnn import PixelCNN
from pixelcnn import MaskedConv2d
from utils import (
    clear_checkpoint,
    clear_log,
    get_last_checkpoint_step,
    ignore_param,
    init_out_dir,
    my_log,
    print_args,
)

import time;


class VAN(nn.Module):
    def __init__(self, L, depth, width, kernel_rad, device, dilations=None):
        """
        Parameters:
        -----------
        L: int, system size,
        depth: int, number of convolutional layers,
        width: int, number of channels in hidden layers,
        kernel_rad: int, radius of the convolutional kernels, 
        device: 'cpu' or 'cuda',
        dilations (optional): list of dilations for each convolution
        """
        super().__init__();
        self.depth = depth;
        self.width = width;
        self.kernel_rad = kernel_rad;
        ks = 2*kernel_rad+1;
        self.ks = ks;
        if dilations is None:
            dilations = 2**np.arange(depth);
        self.dilations = dilations;

        self.epsilon = 1e-7
        self.device = device;
        self.L = L;
        layers = [];
        for i in range(depth):
            layers.append(nn.PReLU());
            if i == 0:
                layers.append(MaskedConv2d(1, width, [ks, ks], padding='same', dilation=self.dilations[i], exclusive=True));
            elif i == depth - 1:
                layers.append(MaskedConv2d(width, 1, [ks, ks], padding='same', dilation=self.dilations[i], exclusive=False));
            else:
                layers.append(MaskedConv2d(width, width, [ks, ks], padding='same', dilation=self.dilations[i], exclusive=False));
        layers.append(nn.Sigmoid());
        self.net = nn.Sequential(*layers);
    
    def forward(self, x):
        return self.net(x);
    
    def sample(self, batch_size):
        #TODO: evaluate only relevant output neuron
        sample = torch.zeros(
            [batch_size, 1, self.L, self.L],
            device=self.device)
        for i in range(self.L):
            for j in range(self.L):
                x_hat = self.forward(sample)
                sample[:, :, i, j] = torch.bernoulli(
                    x_hat[:, :, i, j]) * 2 - 1
        
        return sample, x_hat
    
    def _log_prob(self, sample, x_hat):
        mask = (sample + 1) / 2
        log_prob = (torch.log(x_hat + self.epsilon) * mask +
                    torch.log(1 - x_hat + self.epsilon) * (1 - mask))
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob
    
    def log_prob(self, sample):
        x_hat = self.forward(sample)
        log_prob = self._log_prob(sample, x_hat)
        return log_prob
    
    def fast_sampling(self, batch_size):
        """Generates samples

        Parameters:
        -----------
        batch_size: int, number of samples to produce

        Returns:
        --------
        samples: tensor of shape (batch_size, 1, L, L),
        xhat: tensor of shape (batch_size, 1, L, L), prob. for each spin pointing up

        Notes:
        ------
        same as VAN.sample, but much more efficient. The code is messy, but yields same results as VAN.sample if identical random seeds are chosen."""
        #torch.manual_seed(0);
        samples = torch.zeros([self.L, self.L, 1, batch_size], device=self.device);
        xhat = torch.zeros([self.L, self.L, 1, batch_size], device=self.device);
        temp = torch.zeros([self.L, self.L, self.depth-1, self.width, batch_size], device=self.device); #to try for further optimization: swap axes
        #torch.zeros([batch_size, self.L, self.L], device=self.device);
        with torch.no_grad():
            for i in range(self.L):
                for j in range(self.L):
                    
                    #first layer
                    dil = self.net[1].dilation[0];
                    weightl = self.net[1].weight.data;
                    bias = self.net[1].bias.data;
                    #result = torch.zeros([batch_size, self.width], device=self.device);
                    for ik in range(-self.kernel_rad, 1):
                        for jk in range(-self.kernel_rad, self.kernel_rad+1):
                            #check if spin is already determined and not out off range
                            if (ik < 0 or (ik == 0 and jk < 0)) \
                            and ((i + ik*dil) >= 0) and ((j + jk*dil) >= 0) and ((j + jk*dil) < self.L):
                                mid = self.kernel_rad;
                                temp[i,j,0,:,:] += torch.matmul(weightl[:,:,mid+ik,mid+jk], self.net[0](samples[i+ik*dil,j+jk*dil,:,:]));
                    temp[i,j,0,:,:] += bias[:,None];
                    
                    #following layers
                    for l in range(1, self.depth-1):
                        dil = self.net[2*l+1].dilation[0];
                        weightl = self.net[2*l+1].weight.data;
                        bias = self.net[2*l+1].bias.data;
                        #result = torch.zeros([batch_size, self.width], device=self.device);
                        for ik in range(-self.kernel_rad, 1):
                            for jk in range(-self.kernel_rad, self.kernel_rad+1):
                                #check if spin is already determined and not out off range
                                if (ik < 0 or (ik == 0 and jk <= 0)) \
                                and ((i + ik*dil) >= 0) and ((j + jk*dil) >= 0) and ((j + jk*dil)) < self.L:
                                    mid = self.kernel_rad;
                                    temp[i,j,l,:,:] += torch.matmul(weightl[:,:,mid+ik,mid+jk], self.net[2*l](temp[i+ik*dil,j+jk*dil,l-1,:,:]));
                        temp[i,j,l,:,:] += bias[:,None];                                                 
                    
                    #last layer
                    l = self.depth-1;
                    dil = self.net[2*l+1].dilation[0];
                    weightl = self.net[2*self.depth-1].weight.data;
                    bias = self.net[2*self.depth-1].bias.data;
                    for ik in range(-self.kernel_rad, 1):
                            for jk in range(-self.kernel_rad, self.kernel_rad+1):
                                #check if spin is already determined and not out off range
                                if (ik < 0 or (ik == 0 and jk <= 0)) \
                                and ((i + ik*dil) >= 0) and ((j + jk*dil) >= 0) and ((j + jk*dil)) < self.L:
                                    mid = self.kernel_rad;
                                    xhat[i,j,:,:] += torch.matmul(weightl[:,:,mid+ik,mid+jk], self.net[2*l](temp[i+ik*dil,j+jk*dil,l-1,:,:]));
                    xhat[i,j,:,:] += bias[:,None];
                    xhat[i,j,:,:] = self.net[-1](xhat[i,j,:,:]); #apply tanh
                    
                    #choice
                    samples[i,j,:,:] = torch.bernoulli(xhat[i,j,:,:].T).T * 2 - 1;
                    
        return samples.transpose(0,2).transpose(1,3).transpose(0,1), xhat.transpose(0,2).transpose(1,3).transpose(0,1);

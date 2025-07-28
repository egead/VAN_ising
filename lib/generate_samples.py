import numpy as np
import torch
import torchvision
from numpy import sqrt
from torch import nn

import ising;
from VAN import VAN;


def generate_samples(net, N_samples, N_sampling_batch, keep_samples=True):
    """
    Generates samples of a model with proposal probabilities

    Parameters
    ----------
    net : Model (of class VAN)
    N_samples : number of samples
    N_sampling_batch : number of samples per batch, has to be chosen such that the avaiable (GPU-)memory is sufficient.
    keep_samples : wether to save the samples, default=True (turn off for memory concerns)
    
    Returns
    -------
    samples : np-array of shape (N_samples, net.L, net.L) if keep_samples
    energy : np-array of shape (N_samples,) with energy per spin for each sample
    log_prob : np-array of shape (N_samples,) with proposal log-probability for each sample
    """
    if keep_samples:
        samples = np.zeros((N_samples, net.L, net.L));
    log_prob = np.zeros(N_samples);
    energy = np.zeros(N_samples);
    
    for i in range(N_samples // N_sampling_batch):

        samples_cu, xhat_cu = net.fast_sampling(N_sampling_batch);
        
        if keep_samples:
            samples[i*N_sampling_batch:(i+1)*N_sampling_batch] = samples_cu.view(-1,net.L,net.L).cpu().numpy();
        
        energy[i*N_sampling_batch:(i+1)*N_sampling_batch] = ising.energy(samples_cu, 'fm', 'sqr', 'periodic').cpu().numpy() / net.L**2;
        
        mask = (samples_cu + 1) / 2;
        prob = xhat_cu * mask + (1 - xhat_cu) * (1 - mask);
        log_prob[i*N_sampling_batch:(i+1)*N_sampling_batch] = torch.sum(torch.log(prob.view(-1,net.L,net.L)), dim=(1,2)).cpu().numpy();
        
    
    if keep_samples:
        return samples, energy, log_prob;
    else:
        return energy, log_prob;

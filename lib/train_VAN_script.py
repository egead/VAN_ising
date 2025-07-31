import torch
import sys
sys.path.append('../lib')
import matplotlib.pyplot as plt
import time
import args
import ising  
from VAN import VAN  

args.beta = 0.44
args.L = 16
args.max_step = 10000  #number of epochs
args.batch_size = 1000
args.lr = 1e-3  #learning rate

def train(net, epochs=args.max_step, batch_size=args.batch_size, learning_rate=args.lr, beta=args.beta):
    time_start = time.time()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    for i in range(epochs):
        optimizer.zero_grad()
        
        with torch.no_grad():
            sample, x_hat = net.fast_sampling(batch_size) 
        
        with torch.no_grad():
            energy = ising.energy(sample, 'fm', 'sqr', 'periodic')  
        
    
        log_prob = net.log_prob(sample)  # calculate proposal probability
        
        loss = log_prob - beta * energy  # -beta * energy is target log-probability (this was + before)
        
        # REINFORCE objective (optimize Kullback-Leibler-Divergence)
        baseline = loss.mean()  
        loss_reinforce = -torch.mean((loss - baseline) * log_prob) #CHANGED THE SIGN HERE
        loss_reinforce.backward()
        optimizer.step()
        
        if i % 100 == 0:
            delta_time = time.time() - time_start
            time_start = time.time()
            print(f"Epoch {i}, Time: {delta_time:.2f}s, Loss: {loss_reinforce.item():.4f}")

# Initialize the network (replace 'cuda' with 'cpu' if necessary)
net = VAN(args.L, 4, 64, 1, 'cuda').to('cuda')
train(net)
torch.save(net.state_dict(), 'L16_VAN_10000_BETA044.pt')
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../lib')
import args
from VAN import VAN
from nmcmc_autocorr import plot_autocorrel_function, fitting_tau, autocorr
import ising

def nmcmc_sample_sequence(net, n_samples, beta, ham='fm', lattice='sqr', boundary='periodic'):
    L = net.L
    device = next(net.parameters()).device
    
    current_config = torch.randint(0, 2, (1, 1, L, L), device=device, dtype=torch.float32) * 2 - 1
    current_energy = ising.energy(current_config, ham, lattice, boundary).item()
    current_log_prob = net.log_prob(current_config).item()
    
    samples = []
    energies = []
    log_probs = []
    accept_count = 0
    
    for step in range(n_samples):
        if step % 1000 == 0:
            print(f"Step {step}/{n_samples}, acceptance rate: {accept_count/(step+1e-6):.3f}")
        
        with torch.no_grad():
            if hasattr(net, 'fast_sampling'):
                proposal_config, proposal_log_prob = net.fast_sampling(1)
            else:
                proposal_config, proposal_log_prob = net.sample(1)
        
        proposal_energy = ising.energy(proposal_config, ham, lattice, boundary).item()
        proposal_log_prob = proposal_log_prob.sum().item()
        
        log_accept_prob = (current_log_prob - proposal_log_prob + 
                          beta * (current_energy - proposal_energy))
        log_accept_prob = min(log_accept_prob, 0)
        accept_prob = np.exp(log_accept_prob)
        
        if np.random.rand() < accept_prob:
            current_config = proposal_config.clone()
            current_energy = proposal_energy
            current_log_prob = proposal_log_prob
            accept_count += 1
        
        samples.append(current_config.clone())
        energies.append(current_energy)
        log_probs.append(current_log_prob)
    
    samples = torch.cat(samples, dim=0)
    energies = np.array(energies)
    log_probs = np.array(log_probs)
    
    return samples, energies, log_probs, accept_count / n_samples

def analyze_nmcmc_autocorr(samples, energies, L):
    energies_per_site = energies / (L**2)
    mags = samples.mean(dim=(1,2,3)).cpu().numpy()
    
    hist_len = min(100, len(energies)//4)
    gamma_E, err_gamma_E = autocorr(energies_per_site, hist=hist_len)
    gamma_M, err_gamma_M = autocorr(mags, hist=hist_len)
    
    results = {
        'gamma_E': gamma_E, 'err_gamma_E': err_gamma_E,
        'gamma_M': gamma_M, 'err_gamma_M': err_gamma_M,
        'energies': energies_per_site, 'mags': mags
    }
    
    fitting_range = [2, min(20, len(energies)//20)]
    
    _, tau_E, tau_E_err = fitting_tau(fitting_range, gamma_E, err_gamma_E)
    results['tau_E'] = tau_E
    results['tau_E_err'] = tau_E_err
    
    _, tau_M, tau_M_err = fitting_tau(fitting_range, gamma_M, err_gamma_M)
    results['tau_M'] = tau_M
    results['tau_M_err'] = tau_M_err
    
    return results


args.beta = 0.44
args.L = 16

net = VAN(args.L, 4, 64, 1, 'cuda').to('cuda')
net.load_state_dict(torch.load('L16_VAN_10000_BETA044_new.pt', weights_only=True))

n_samples = 500000
samples, energies, log_probs, acceptance_rate = nmcmc_sample_sequence(
    net, n_samples, args.beta, 'fm', 'sqr', 'periodic'
)

print(f"Mean energy per site: {energies.mean()/(args.L**2):.3f}")
print(f"Mean magnetization: {samples.mean().item():.3f}")
print(f"Acceptance rate: {acceptance_rate:.3f}")

results = analyze_nmcmc_autocorr(samples, energies, args.L)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(results['gamma_E'][:50], 'b.-')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Energy Autocorrelation')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(results['gamma_M'][:50], 'r.-')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Magnetization Autocorrelation')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("nmcmc_autocorr.png", dpi=300, bbox_inches='tight')
plt.show()

fitting_range = [2, min(20, n_samples//20)]

params_E, _, _ = fitting_tau(fitting_range, results['gamma_E'], results['err_gamma_E'])
plot_autocorrel_function(fitting_range, params_E, results['gamma_E'], 
                        results['err_gamma_E'], save_name="energy_fit.png")

params_M, _, _ = fitting_tau(fitting_range, results['gamma_M'], results['err_gamma_M'])
plot_autocorrel_function(fitting_range, params_M, results['gamma_M'], 
                        results['err_gamma_M'], save_name="mag_fit.png")

print(f"Energy tau: {results['tau_E']:.2f} ± {results['tau_E_err']:.2f}")
print(f"Magnetization tau: {results['tau_M']:.2f} ± {results['tau_M_err']:.2f}")
import torch
import numpy as np
from nmcmc_analyzer import NMCMCAnalyzer
from VAN import VAN
import ising
torch.cuda.empty_cache()
def ising_energy(configs):
    """Compute Ising energy using the same function as training"""
    return ising.energy(configs, 'fm', 'sqr', 'periodic')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
beta = 0.44
L = 16
batch_size = 32
n_steps = 1000

van_model = VAN(L, 4, 64, 1, device)
state_dict = torch.load('L16_VAN_10000_BETA044.pt', map_location=device, weights_only=True)
van_model.load_state_dict(state_dict)
van_model.to(device)
van_model.eval()

print("Testing VAN model...")
with torch.no_grad():
    samples, _ = van_model.sample(100)
    energies = ising_energy(samples)
    print(f"VAN sample energies - mean: {energies.mean():.3f}, std: {energies.std():.3f}")
    
    # Compare to random samples
    random_configs = torch.randint(0, 2, (100, 1, L, L), dtype=torch.float32) * 2 - 1
    random_energies = ising_energy(random_configs.to(device))
    print(f"Random energies - mean: {random_energies.mean():.3f}, std: {random_energies.std():.3f}")

initial_config = torch.randint(0, 2, (batch_size, 1, L, L), dtype=torch.float32) * 2 - 1
analyzer = NMCMCAnalyzer(van_model, ising_energy, beta, device)

print("Running NMCMC...")
mcmc_results = analyzer.run_nmcmc(initial_config, n_steps)

print("NMCMC diagnostics:")
print(f"Acceptance rate: {mcmc_results['acceptance_rate']:.6f}")
importance_ratios = mcmc_results['importance_ratios']
print(f"Importance ratios - min: {importance_ratios.min():.6f}, max: {importance_ratios.max():.6f}")
print(f"Zero ratios: {(importance_ratios == 0).sum()}/{importance_ratios.numel()}")
print(f"Finite ratios: {torch.isfinite(importance_ratios).sum()}/{importance_ratios.numel()}")

configs = mcmc_results['configs']
configs_flat = configs.view(configs.shape[0], -1)
config_autocorr = analyzer.compute_autocorrelation(configs_flat.mean(dim=1))
tau_int, _ = analyzer.estimate_integrated_time(config_autocorr)

print(f"Acceptance rate: {mcmc_results['acceptance_rate']:.3f}")
print(f"Configuration τ_int: {tau_int:.2f}")

batch_result = analyzer.compute_batch_eigenvalue(mcmc_results['importance_ratios'])
print(f"General mixing τ_int_B: {batch_result['tau_int_B']:.2f}")
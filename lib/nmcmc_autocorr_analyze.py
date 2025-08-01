#new script
import torch
import sys
sys.path.append('../lib')

import args
from VAN import VAN
from nmcmc_autocorr import sample_van_sequence, analyze_van_autocorr, plot_autocorrel_function, fitting_tau

args.beta = 0.44
args.L = 16

net = VAN(args.L, 4, 64, 1, 'cuda').to('cuda')
net.load_state_dict(torch.load('L16_VAN_10000_BETA044_new.pt', weights_only=True))

n_samples = 5000
samples = sample_van_sequence(net, n_samples)
results = analyze_van_autocorr(samples, 'fm', 'sqr', 'periodic', args.L)

fitting_range = [2, min(50, n_samples//8)]
if not torch.isnan(torch.tensor(results['tau_E'])):
    params_E, _, _ = fitting_tau(fitting_range, results['gamma_E'], results['err_gamma_E'])
    plot_autocorrel_function(fitting_range, params_E, results['gamma_E'], results['err_gamma_E'])

print(f"Energy tau: {results['tau_E']:.2f} ± {results['tau_E_err']:.2f}")
print(f"Magnetization tau: {results['tau_M']:.2f} ± {results['tau_M_err']:.2f}")
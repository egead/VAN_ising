import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Callable
import warnings


class NMCMCAnalyzer:
    """
    Neural Markov Chain Monte Carlo analyzer for Variational Autoregressive Networks.
    
    Implements autocorrelation analysis following:
    "Analysis of autocorrelation times in Neural Markov Chain Monte Carlo simulations"
    by Białas et al. (2023)
    """
    
    def __init__(self, van_model, energy_function: Callable, beta: float, device: str = 'cuda'):
        """
        Initialize NMCMC analyzer.
        
        Args:
            van_model: Trained VAN model with sample() and log_prob() methods
            energy_function: Function that computes energy E(s) for configuration s
            beta: Inverse temperature β = 1/T
            device: Computing device ('cuda' or 'cpu')
        """
        self.van_model = van_model
        self.energy_function = energy_function
        self.beta = beta
        self.device = device
        self.van_model.eval()
        
    def run_nmcmc(self, initial_config: torch.Tensor, n_steps: int, 
                   n_patches_per_step: int = 1) -> Dict:
        """
        Run Neural MCMC sampling using VAN proposals.
        
        Args:
            initial_config: Initial configuration tensor
            n_steps: Number of MCMC steps
            n_patches_per_step: Number of patch proposals per step (for enhanced sampling)
            
        Returns:
            Dictionary containing:
                - configs: Tensor of configurations [n_steps, *config_shape]
                - energies: Array of energies per spin
                - log_probs: Array of VAN log probabilities
                - importance_ratios: Array of importance ratios w(s) = p(s)/q(s)
                - acceptance_rate: Overall acceptance rate
        """
        config = initial_config.clone().to(self.device)
        batch_size = config.shape[0]
        
        configs = []
        energies = []
        log_probs = []
        importance_ratios = []
        n_accepted = 0
        n_total = 0
        
        # Initial state
        current_energy = self.energy_function(config)
        current_log_prob = self.van_model.log_prob(config)
        
        for step in range(n_steps):
            step_config = config.clone()
            
            # Multiple proposals per step (for enhanced sampling)
            for _ in range(n_patches_per_step):
                # Generate proposal from VAN
                proposal, proposal_xhat = self.van_model.sample(batch_size)
                proposal_energy = self.energy_function(proposal)
                proposal_log_prob = self.van_model.log_prob(proposal)
                
                # Compute acceptance probability
                # For NMCMC: min(1, p(s_new)q(s_old) / (p(s_old)q(s_new)))
                # p(s) ∝ exp(-βE(s)), q(s) from VAN
                delta_log_prob = (proposal_log_prob - current_log_prob)
                delta_energy = proposal_energy - current_energy
                
                log_alpha = -self.beta * delta_energy + delta_log_prob
                alpha = torch.exp(torch.clamp(log_alpha, max=0))
                
                # Accept or reject
                accept_mask = torch.rand(batch_size, device=self.device) < alpha
                
                # Update accepted configurations
                if accept_mask.any():
                    step_config[accept_mask] = proposal[accept_mask]
                    current_energy[accept_mask] = proposal_energy[accept_mask]
                    current_log_prob[accept_mask] = proposal_log_prob[accept_mask]
                    n_accepted += accept_mask.sum().item()
                
                n_total += batch_size
            
            # Store configuration
            configs.append(step_config.cpu().clone())
            energies.append(current_energy.cpu().clone())
            log_probs.append(current_log_prob.cpu().clone())
            
            # FIXED: Compute importance ratios w(s) = p(s)/q(s) = exp(-βE(s))/q(s)
            # Use log-space for numerical stability
            log_importance = -self.beta * current_energy - current_log_prob
            # Normalize by subtracting max for numerical stability
            log_importance = log_importance - torch.max(log_importance)
            importance = torch.exp(log_importance)
            importance_ratios.append(importance.cpu().clone())
            
            config = step_config
            
            # Progress reporting
            if (step + 1) % max(1, n_steps // 10) == 0:
                acc_rate = n_accepted / n_total if n_total > 0 else 0
                print(f"Step {step+1}/{n_steps}, Acceptance rate: {acc_rate:.3f}")
        
        # Convert to arrays/tensors
        configs = torch.stack(configs, dim=0)  # [n_steps, batch_size, ...]
        # FIXED: Remove per-spin normalization - keep total energy
        energies = torch.stack(energies, dim=0)  # Total energy
        log_probs = torch.stack(log_probs, dim=0)
        importance_ratios = torch.stack(importance_ratios, dim=0)
        
        return {
            'configs': configs,
            'energies': energies,
            'log_probs': log_probs,
            'importance_ratios': importance_ratios,
            'acceptance_rate': n_accepted / n_total if n_total > 0 else 0
        }
    
    def compute_autocorrelation(self, observable_chain: np.ndarray, 
                               max_lag: Optional[int] = None) -> np.ndarray:
        """
        Compute autocorrelation function Γ_O(t).
        
        Γ_O(t) = <(O(s_i) - <O>)(O(s_{i+t}) - <O>)> / Var(O)
        
        Args:
            observable_chain: Time series of observable values [n_steps, batch_size]
            max_lag: Maximum lag to compute (default: n_steps//4)
            
        Returns:
            Autocorrelation function Γ(t) for t = 0, 1, ..., max_lag
        """
        if isinstance(observable_chain, torch.Tensor):
            observable_chain = observable_chain.cpu().numpy()
        
        # Average over batch dimension if present
        if observable_chain.ndim > 1:
            observable_chain = np.mean(observable_chain, axis=1)
        
        n_steps = len(observable_chain)
        if max_lag is None:
            max_lag = min(n_steps // 4, 200)  # Reasonable default
        
        # Center the data
        mean_obs = np.mean(observable_chain)
        centered_obs = observable_chain - mean_obs
        var_obs = np.var(observable_chain, ddof=1)
        
        if var_obs == 0:
            return np.ones(max_lag + 1)  # Constant observable
        
        # Compute autocorrelation using numpy correlate
        autocorr = np.zeros(max_lag + 1)
        for t in range(max_lag + 1):
            if t == 0:
                autocorr[t] = 1.0
            else:
                if n_steps - t > 0:
                    autocorr[t] = np.mean(centered_obs[:-t] * centered_obs[t:]) / var_obs
                else:
                    autocorr[t] = 0.0
        
        return autocorr
    
    def estimate_integrated_time(self, autocorr_function: np.ndarray) -> Tuple[float, int]:
        """
        Estimate integrated autocorrelation time τ^{O,int}.
        
        τ^{O,int} = 1 + 2 * Σ_{t=1}^{t_max} Γ_O(t)
        
        Args:
            autocorr_function: Autocorrelation function Γ(t)
            
        Returns:
            Tuple of (integrated_time, t_max_used)
        """
        # Find t_max where autocorrelation first becomes negative or very small
        t_max = len(autocorr_function) - 1
        for t in range(1, len(autocorr_function)):
            if autocorr_function[t] <= 0 or autocorr_function[t] < 0.01:
                t_max = t - 1
                break
        
        if t_max <= 0:
            return 1.0, 0
        
        # Compute integrated time
        integrated_time = 1.0 + 2.0 * np.sum(autocorr_function[1:t_max+1])
        
        return max(integrated_time, 1.0), t_max
    
    def fit_exponential_decay(self, autocorr_function: np.ndarray, 
                             fit_range: Optional[Tuple[int, int]] = None) -> Dict:
        """
        Fit exponential decay to autocorrelation function.
        
        Fits: Γ(t) ~ a₁*exp(-t/τ₁) + a₂*exp(-t/τ₂)
        
        Args:
            autocorr_function: Autocorrelation function Γ(t)
            fit_range: Range (t_start, t_end) for fitting, auto-determined if None
            
        Returns:
            Dictionary with fit parameters and τ^{O,int}_F
        """
        if fit_range is None:
            # Auto-determine fit range (skip very early times, stop when Γ becomes small)
            t_start = max(1, len(autocorr_function) // 10)
            t_end = len(autocorr_function) - 1
            for t in range(t_start, len(autocorr_function)):
                if autocorr_function[t] <= 0.01 or autocorr_function[t] <= 0:
                    t_end = t
                    break
        else:
            t_start, t_end = fit_range
        
        t_fit = np.arange(t_start, min(t_end, len(autocorr_function)))
        if len(t_fit) < 5:  # Need minimum points for fitting
            return {'tau_F': 1.0, 'tau_int_F': 1.0, 'fit_success': False}
        
        gamma_fit = autocorr_function[t_fit]
        
        # Remove non-positive values
        valid_mask = gamma_fit > 0
        if np.sum(valid_mask) < 3:
            return {'tau_F': 1.0, 'tau_int_F': 1.0, 'fit_success': False}
        
        t_fit = t_fit[valid_mask]
        gamma_fit = gamma_fit[valid_mask]
        
        try:
            # Try two-exponential fit first
            def two_exp(t, a1, tau1, a2, tau2):
                return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)
            
            # Initial guess
            p0 = [gamma_fit[0] * 0.6, 2.0, gamma_fit[0] * 0.4, 10.0]
            bounds = ([0, 0.1, 0, 0.1], [np.inf, 1000, np.inf, 1000])
            
            popt, _ = curve_fit(two_exp, t_fit, gamma_fit, p0=p0, bounds=bounds)
            tau_F = max(popt[1], popt[3])
            fit_success = True
            
        except:
            try:
                # Fallback to single exponential
                def single_exp(t, a, tau):
                    return a * np.exp(-t / tau)
                
                p0 = [gamma_fit[0], 5.0]
                bounds = ([0, 0.1], [np.inf, 1000])
                
                popt, _ = curve_fit(single_exp, t_fit, gamma_fit, p0=p0, bounds=bounds)
                tau_F = popt[1]
                fit_success = True
                
            except:
                # Ultimate fallback: estimate from decay
                if len(gamma_fit) > 1:
                    tau_F = -1.0 / np.mean(np.diff(np.log(gamma_fit + 1e-10)))
                    tau_F = max(tau_F, 1.0)
                else:
                    tau_F = 1.0
                fit_success = False
        
        # Compute integrated version
        tau_int_F = (1 + np.exp(-1.0 / tau_F)) / (1 - np.exp(-1.0 / tau_F))
        
        return {
            'tau_F': tau_F, 
            'tau_int_F': tau_int_F,
            'fit_success': fit_success,
            'fit_range': (t_start, t_end)
        }
    
    def compute_batch_eigenvalue(self, importance_ratios: torch.Tensor) -> Dict:
        """
        Compute batch eigenvalue estimator τ^{int}_B.
        
        Following Eq. (22-24) from Białas et al.:
        λ̂₁ = (1/N_batch) * Σ_k (1 - w(s_k)/w_max)
        τ̂_B = -1/log(λ̂₁)
        
        Args:
            importance_ratios: Importance ratios w(s) = p(s)/q(s) [n_steps, batch_size]
            
        Returns:
            Dictionary with eigenvalue estimates
        """
        if isinstance(importance_ratios, torch.Tensor):
            w = importance_ratios.cpu().detach().numpy()
        else:
            w = importance_ratios
        
        # Flatten batch dimensions
        w_flat = w.flatten()
        w_max = np.max(w_flat)
        
        # FIXED: Add safety check for zero w_max
        if w_max <= 0:
            return {'lambda_1': 0.0, 'tau_B': np.inf, 'tau_int_B': np.inf}
        
        # Compute λ̂₁
        lambda_1 = np.mean(1.0 - w_flat / w_max)
        lambda_1 = np.clip(lambda_1, 1e-10, 1.0 - 1e-10)  # Numerical stability
        
        # Compute τ̂_B
        tau_B = -1.0 / np.log(lambda_1)
        
        # Compute integrated version
        tau_int_B = (1 + np.exp(-1.0 / tau_B)) / (1 - np.exp(-1.0 / tau_B))
        
        return {
            'lambda_1': lambda_1,
            'tau_B': tau_B,
            'tau_int_B': tau_int_B,
            'w_max': w_max,
            'w_mean': np.mean(w_flat)
        }
    
    def analyze_observables(self, mcmc_results: Dict, observables: Optional[Dict] = None) -> Dict:
        """
        Comprehensive autocorrelation analysis for multiple observables.
        
        Args:
            mcmc_results: Results from run_nmcmc()
            observables: Dict of {name: observable_function} or None for defaults
            
        Returns:
            Dictionary with autocorrelation analysis results
        """
        configs = mcmc_results['configs']
        energies = mcmc_results['energies']
        importance_ratios = mcmc_results['importance_ratios']
        
        # Default observables
        if observables is None:
            observables = {
                'energy': lambda x: energies,
                'magnetization': self._compute_magnetization
            }
        
        results = {}
        
        # Analyze each observable
        for obs_name, obs_func in observables.items():
            print(f"Analyzing observable: {obs_name}")
            
            if obs_name == 'energy':
                obs_chain = energies
            else:
                obs_chain = obs_func(configs)
            
            # Compute autocorrelation function
            autocorr = self.compute_autocorrelation(obs_chain)
            
            # Estimate integrated time
            tau_int, t_max = self.estimate_integrated_time(autocorr)
            
            # Fit exponential decay
            exp_fit = self.fit_exponential_decay(autocorr)
            
            results[obs_name] = {
                'autocorrelation': autocorr,
                'tau_int': tau_int,
                'tau_int_F': exp_fit['tau_int_F'],
                'tau_F': exp_fit['tau_F'],
                't_max_used': t_max,
                'exp_fit_success': exp_fit['fit_success'],
                'observable_mean': np.mean(obs_chain.cpu().numpy() if isinstance(obs_chain, torch.Tensor) else obs_chain),
                'observable_std': np.std(obs_chain.cpu().numpy() if isinstance(obs_chain, torch.Tensor) else obs_chain)
            }
        
        # Compute batch eigenvalue estimator
        batch_results = self.compute_batch_eigenvalue(importance_ratios)
        results['batch_eigenvalue'] = batch_results
        
        # Summary statistics
        results['summary'] = {
            'acceptance_rate': mcmc_results['acceptance_rate'],
            'n_steps': configs.shape[0],
            'batch_size': configs.shape[1],
            'beta': self.beta
        }
        
        return results
    
    def _compute_magnetization(self, configs: torch.Tensor) -> torch.Tensor:
        """Compute absolute magnetization per spin."""
        # configs shape: [n_steps, batch_size, L, L]
        magnetization = torch.abs(torch.sum(configs, dim=(-2, -1))) / configs.shape[-1]**2
        return magnetization
    
    def plot_autocorrelation_analysis(self, analysis_results: Dict, 
                                     save_path: Optional[str] = None):
        """
        Plot comprehensive autocorrelation analysis results.
        
        Args:
            analysis_results: Results from analyze_observables()
            save_path: Path to save plots (optional)
        """
        observables = {k: v for k, v in analysis_results.items() 
                      if k not in ['batch_eigenvalue', 'summary']}
        
        n_obs = len(observables)
        fig, axes = plt.subplots(2, n_obs, figsize=(5*n_obs, 10))
        if n_obs == 1:
            axes = axes.reshape(2, 1)
        
        for i, (obs_name, obs_data) in enumerate(observables.items()):
            # Plot autocorrelation function
            autocorr = obs_data['autocorrelation']
            t_array = np.arange(len(autocorr))
            
            axes[0, i].semilogy(t_array, np.maximum(autocorr, 1e-6), 'b-', 
                               label=f'Γ(t)', linewidth=2)
            axes[0, i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[0, i].set_xlabel('Time lag t')
            axes[0, i].set_ylabel('Autocorrelation Γ(t)')
            axes[0, i].set_title(f'Autocorrelation: {obs_name}')
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend()
            
            # Add text with results
            info_text = f"τ_int = {obs_data['tau_int']:.2f}\n"
            info_text += f"τ_int_F = {obs_data['tau_int_F']:.2f}\n"
            info_text += f"τ_F = {obs_data['tau_F']:.2f}"
            axes[0, i].text(0.05, 0.95, info_text, transform=axes[0, i].transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='wheat', alpha=0.8))
            
            # Plot comparison of different estimators
            estimators = ['tau_int', 'tau_int_F']
            values = [obs_data[est] for est in estimators]
            
            axes[1, i].bar(estimators, values, alpha=0.7)
            axes[1, i].set_ylabel('Autocorrelation Time')
            axes[1, i].set_title(f'Estimator Comparison: {obs_name}')
            axes[1, i].set_yscale('log')
            
            # Add batch eigenvalue result if available
            if 'batch_eigenvalue' in analysis_results:
                batch_tau = analysis_results['batch_eigenvalue']['tau_int_B']
                axes[1, i].axhline(y=batch_tau, color='red', linestyle='--', 
                                  label=f'τ_int_B = {batch_tau:.2f}')
                axes[1, i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print summary
        print("\n" + "="*60)
        print("AUTOCORRELATION ANALYSIS SUMMARY")
        print("="*60)
        print(f"Acceptance rate: {analysis_results['summary']['acceptance_rate']:.3f}")
        print(f"Number of steps: {analysis_results['summary']['n_steps']}")
        print(f"Temperature β: {analysis_results['summary']['beta']:.3f}")
        
        if 'batch_eigenvalue' in analysis_results:
            batch_data = analysis_results['batch_eigenvalue']
            print(f"\nBatch eigenvalue estimator:")
            print(f"  λ₁ = {batch_data['lambda_1']:.4f}")
            print(f"  τ_B = {batch_data['tau_B']:.2f}")
            print(f"  τ_int_B = {batch_data['tau_int_B']:.2f}")
        
        print(f"\nObservable-specific results:")
        for obs_name, obs_data in observables.items():
            print(f"  {obs_name}:")
            print(f"    τ_int = {obs_data['tau_int']:.2f}")
            print(f"    τ_int_F = {obs_data['tau_int_F']:.2f}")
            print(f"    Mean = {obs_data['observable_mean']:.4f}")
            print(f"    Std = {obs_data['observable_std']:.4f}")
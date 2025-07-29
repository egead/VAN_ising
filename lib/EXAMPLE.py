import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, Optional

# Import the VAN class from your project files
# Assuming VAN.py is in your path
from VAN import VAN
from AUTOCORR import NMCMCAnalyzer
# Import the NMCMCAnalyzer from the previous implementation
# (Copy the NMCMCAnalyzer class from the paste.txt content)


def load_van_model(model_path: str, L: int, depth: int, width: int, 
                   kernel_rad: int, device: str = 'cuda') -> VAN:
    """
    Load a saved VAN model.
    
    Args:
        model_path: Path to the saved VAN.pt file
        L: Lattice size
        depth: Network depth
        width: Network width  
        kernel_rad: Kernel radius
        device: Device to load on
        
    Returns:
        Loaded VAN model
    """
    # Initialize VAN with same parameters as training
    van_model = VAN(L=L, depth=depth, width=width, 
                    kernel_rad=kernel_rad, device=device)
    
    # Load the saved state dict
    van_model.load_state_dict(torch.load(model_path, map_location=device))
    van_model.eval()
    
    print(f"Loaded VAN model from {model_path}")
    print(f"Model parameters: L={L}, depth={depth}, width={width}, kernel_rad={kernel_rad}")
    
    return van_model


def ising_energy_2d(configs: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D Ising energy with periodic boundary conditions.
    
    Args:
        configs: Configuration tensor [batch_size, 1, L, L] or [batch_size, L, L]
        
    Returns:
        Energy tensor [batch_size]
    """
    if configs.dim() == 3:
        configs = configs.unsqueeze(1)  # Add channel dimension
    
    batch_size, _, L, _ = configs.shape
    
    # Horizontal nearest neighbor interactions
    horizontal = configs[:, :, :, :-1] * configs[:, :, :, 1:]  # [batch, 1, L, L-1]
    horizontal_periodic = configs[:, :, :, -1:] * configs[:, :, :, :1]  # [batch, 1, L, 1]
    
    # Vertical nearest neighbor interactions  
    vertical = configs[:, :, :-1, :] * configs[:, :, 1:, :]  # [batch, 1, L-1, L]
    vertical_periodic = configs[:, :, -1:, :] * configs[:, :, :1, :]  # [batch, 1, 1, L]
    
    # Sum all interactions
    energy = -(horizontal.sum() + horizontal_periodic.sum() + 
              vertical.sum() + vertical_periodic.sum())
    
    # Return per configuration
    energy_per_config = torch.zeros(batch_size, device=configs.device)
    for b in range(batch_size):
        h_sum = (configs[b, 0, :, :-1] * configs[b, 0, :, 1:]).sum()
        h_periodic = (configs[b, 0, :, -1] * configs[b, 0, :, 0]).sum()
        v_sum = (configs[b, 0, :-1, :] * configs[b, 0, 1:, :]).sum()
        v_periodic = (configs[b, 0, -1, :] * configs[b, 0, 0, :]).sum()
        energy_per_config[b] = -(h_sum + h_periodic + v_sum + v_periodic)
    
    return energy_per_config


def basic_nmcmc_example():
    """
    Very basic example of VAN NMCMC analysis.
    """
    
    # =================================================================
    # STEP 1: SETUP PARAMETERS (MODIFY THESE FOR YOUR MODEL)
    # =================================================================
    
    # VAN model parameters (must match your saved model!)
    L = 4              # Lattice size
    depth = 3           # Network depth  
    width = 64          # Network width
    kernel_rad = 1      # Kernel radius
    
    # Physics parameters
    beta = 0.3         # Inverse temperature (near critical for 2D Ising)
    
    # MCMC parameters
    n_steps = 1000      # Number of MCMC steps
    batch_size = 4      # Number of parallel chains
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # =================================================================
    # STEP 2: LOAD YOUR SAVED VAN MODEL
    # =================================================================
    
    model_path = "/home/ege/VAN_Ising/normal_training/L4_VAN_1000epochs.pt"  # PATH TO YOUR SAVED MODEL
    
    try:
        van_model = load_van_model(model_path, L, depth, width, kernel_rad, device)
    except FileNotFoundError:
        print(f"Model file {model_path} not found!")
        print("Creating a mock VAN for demonstration...")
        
        # Create a mock VAN if no saved model (for demo purposes)
        van_model = VAN(L=L, depth=depth, width=width, 
                       kernel_rad=kernel_rad, device=device)
        print("Using untrained VAN model for demonstration")
    
    # =================================================================
    # STEP 3: SETUP NMCMC ANALYZER
    # =================================================================
    
    analyzer = NMCMCAnalyzer(
        van_model=van_model,
        energy_function=ising_energy_2d,
        beta=beta,
        device=device
    )
    
    # =================================================================
    # STEP 4: CREATE INITIAL CONFIGURATION
    # =================================================================
    
    # Random initial configuration
    initial_config = torch.randint(0, 2, (batch_size, 1, L, L), 
                                  dtype=torch.float32, device=device) * 2 - 1
    
    print(f"Initial configuration shape: {initial_config.shape}")
    print(f"Initial energy per spin: {torch.mean(ising_energy_2d(initial_config) / (L*L)):.4f}")
    
    # =================================================================
    # STEP 5: RUN NMCMC SAMPLING
    # =================================================================
    
    print("\n" + "="*50)
    print("RUNNING NEURAL MCMC SAMPLING")
    print("="*50)
    
    mcmc_results = analyzer.run_nmcmc(
        initial_config=initial_config,
        n_steps=n_steps,
        n_patches_per_step=1  # Single proposal per step for simplicity
    )
    
    print(f"\nSampling completed!")
    print(f"Final acceptance rate: {mcmc_results['acceptance_rate']:.3f}")
    
    # =================================================================
    # STEP 6: ANALYZE AUTOCORRELATIONS
    # =================================================================
    
    print("\n" + "="*50)
    print("ANALYZING AUTOCORRELATIONS")
    print("="*50)
    
    analysis_results = analyzer.analyze_observables(mcmc_results)
    
    # =================================================================
    # STEP 7: PLOT RESULTS
    # =================================================================
    
    analyzer.plot_autocorrelation_analysis(analysis_results, 
                                          save_path="basic_nmcmc_analysis.png")
    
    # =================================================================
    # STEP 8: PRINT SUMMARY
    # =================================================================
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    summary = analysis_results['summary']
    print(f"System size: {L}×{L}")
    print(f"Temperature: β = {beta}")
    print(f"MCMC steps: {summary['n_steps']}")
    print(f"Acceptance rate: {summary['acceptance_rate']:.3f}")
    
    # Energy statistics
    final_energies = mcmc_results['energies'][-100:]  # Last 100 steps
    mean_energy = torch.mean(final_energies).item()
    std_energy = torch.std(final_energies).item()
    print(f"Final energy per spin: {mean_energy:.4f} ± {std_energy:.4f}")
    
    # Autocorrelation times
    if 'energy' in analysis_results:
        energy_data = analysis_results['energy']
        print(f"Energy autocorrelation time: {energy_data['tau_int']:.2f}")
    
    if 'magnetization' in analysis_results:
        mag_data = analysis_results['magnetization']
        print(f"Magnetization autocorrelation time: {mag_data['tau_int']:.2f}")
    
    if 'batch_eigenvalue' in analysis_results:
        batch_data = analysis_results['batch_eigenvalue']
        print(f"Batch eigenvalue τ_int_B: {batch_data['tau_int_B']:.2f}")
    
    return analyzer, mcmc_results, analysis_results


def quick_test_run():
    """
    Even simpler test run with minimal output.
    """
    print("Running quick NMCMC test...")
    
    # Create simple VAN
    L, depth, width, kernel_rad = 8, 2, 16, 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    van_model = VAN(L=L, depth=depth, width=width, 
                   kernel_rad=kernel_rad, device=device)
    
    # Setup analyzer
    analyzer = NMCMCAnalyzer(van_model, ising_energy_2d, beta=0.5, device=device)
    
    # Quick run
    initial_config = torch.randint(0, 2, (2, 1, L, L), device=device).float() * 2 - 1
    results = analyzer.run_nmcmc(initial_config, n_steps=100)
    
    print(f"Quick test completed! Acceptance rate: {results['acceptance_rate']:.3f}")
    
    return results


if __name__ == "__main__":
    # Choose which example to run
    run_basic_example = True
    
    if run_basic_example:
        try:
            analyzer, mcmc_results, analysis_results = basic_nmcmc_example()
            print("\nBasic example completed successfully!")
        except Exception as e:
            print(f"Error in basic example: {e}")
            print("Running quick test instead...")
            quick_test_run()
    else:
        quick_test_run()
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import ising
from VAN import VAN

class PatchVAN(VAN):
    """VAN trained specifically for patch generation"""
    def __init__(self, patch_size, depth, width, kernel_rad, device, dilations=None):
        # Initialize with patch size instead of full lattice
        super().__init__(L=patch_size, depth=depth, width=width, 
                        kernel_rad=kernel_rad, device=device, dilations=dilations)
        self.patch_size = patch_size

class EnhancedMetropolis:
    """Enhanced Metropolis sampler using VAN for patch proposals"""
    
    def __init__(self, patch_van: PatchVAN, lattice_size: int, beta: float, device='cuda'):
        """
        Parameters:
        -----------
        patch_van: Trained PatchVAN model for generating patches
        lattice_size: Size of the full lattice (L×L)
        beta: Inverse temperature
        device: 'cpu' or 'cuda'
        """
        self.patch_van = patch_van
        self.L = lattice_size
        self.patch_size = patch_van.patch_size
        self.beta = beta
        self.device = device
        
        # Ensure patch doesn't exceed lattice
        assert self.patch_size <= self.L, "Patch size must be <= lattice size"
        
    def extract_patch_with_boundary(self, config: torch.Tensor, i: int, j: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract a patch centered at (i,j) with boundary conditions
        
        Returns:
        --------
        patch: tensor of shape (1, 1, patch_size, patch_size)
        boundary: tensor of shape (1, 1, L, L) with patch area masked
        """
        # Handle periodic boundaries
        half_p = self.patch_size // 2
        
        # Create indices with periodic wrapping
        i_indices = torch.arange(i - half_p, i - half_p + self.patch_size) % self.L
        j_indices = torch.arange(j - half_p, j - half_p + self.patch_size) % self.L
        
        # Extract patch
        patch = config[:, :, i_indices][:, :, :, j_indices]
        
        # Create boundary mask (full config with patch area zeroed)
        boundary = config.clone()
        for pi in range(self.patch_size):
            for pj in range(self.patch_size):
                bi = (i - half_p + pi) % self.L
                bj = (j - half_p + pj) % self.L
                boundary[:, :, bi, bj] = 0
                
        return patch, boundary
    
    def insert_patch(self, config: torch.Tensor, patch: torch.Tensor, i: int, j: int):
        """Insert patch back into configuration at position (i,j)"""
        half_p = self.patch_size // 2
        
        for pi in range(self.patch_size):
            for pj in range(self.patch_size):
                bi = (i - half_p + pi) % self.L
                bj = (j - half_p + pj) % self.L
                config[:, :, bi, bj] = patch[:, :, pi, pj]
                
        return config
    
    def compute_local_energy_diff(self, config: torch.Tensor, new_patch: torch.Tensor, 
                                 i: int, j: int) -> torch.Tensor:
        """
        Compute energy difference for patch replacement
        Only considers interactions at patch boundary to save computation
        """
        # Create configs with old and new patches
        config_old = config.clone()
        config_new = config.clone()
        config_new = self.insert_patch(config_new, new_patch, i, j)
        
        # Compute full energies (you could optimize this to only compute boundary terms)
        energy_old = ising.energy(config_old, 'fm', 'sqr', 'periodic')
        energy_new = ising.energy(config_new, 'fm', 'sqr', 'periodic')
        
        return energy_new - energy_old
    
    def metropolis_step(self, config: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Single Metropolis step with neural patch proposal
        
        Returns:
        --------
        config: Updated configuration
        accepted: Whether the proposal was accepted
        """
        # Random position for patch center
        i = torch.randint(0, self.L, (1,)).item()
        j = torch.randint(0, self.L, (1,)).item()
        
        # Extract current patch and boundary
        old_patch, boundary = self.extract_patch_with_boundary(config, i, j)
        
        # Generate new patch proposal
        # For now, unconditional generation - you could modify to condition on boundary
        new_patch, _ = self.patch_van.sample(1)
        
        # Compute acceptance probability
        # Energy difference
        delta_E = self.compute_local_energy_diff(config, new_patch, i, j)
        
        # Proposal probability ratio q(old|new) / q(new|old)
        # For symmetric proposals this is 1, for VAN we need actual probabilities
        old_patch_flat = old_patch.view(1, 1, self.patch_size, self.patch_size)
        new_patch_flat = new_patch.view(1, 1, self.patch_size, self.patch_size)
        
        log_q_old = self.patch_van.log_prob(old_patch_flat)
        log_q_new = self.patch_van.log_prob(new_patch_flat)
        
        # Metropolis acceptance ratio
        log_alpha = -self.beta * delta_E + log_q_old - log_q_new
        alpha = torch.exp(torch.clamp(log_alpha, max=0))
        
        # Accept or reject
        if torch.rand(1) < alpha:
            config = self.insert_patch(config, new_patch, i, j)
            accepted = True
        else:
            accepted = False
            
        return config, accepted
    
    def run(self, initial_config: Optional[torch.Tensor] = None, 
            n_steps: int = 1000, 
            n_patches_per_step: int = 10) -> dict:
        """
        Run enhanced Metropolis sampling
        
        Parameters:
        -----------
        initial_config: Starting configuration, random if None
        n_steps: Number of Metropolis steps
        n_patches_per_step: Number of patch updates to attempt per step
        
        Returns:
        --------
        Dictionary with:
            - configs: List of configurations
            - energies: List of energies
            - acceptance_rate: Average acceptance rate
        """
        # Initialize
        if initial_config is None:
            config = torch.randint(0, 2, (1, 1, self.L, self.L), 
                                 device=self.device).float() * 2 - 1
        else:
            config = initial_config.to(self.device)
            
        configs = []
        energies = []
        n_accepted = 0
        n_total = 0
        
        # Main loop
        for step in range(n_steps):
            # Multiple patch updates per step
            for _ in range(n_patches_per_step):
                config, accepted = self.metropolis_step(config)
                n_accepted += accepted
                n_total += 1
            
            # Store configuration
            configs.append(config.cpu().clone())
            energies.append(ising.energy(config, 'fm', 'sqr', 'periodic').cpu().item() / self.L**2)
            
            # Progress
            if step % 100 == 0:
                print(f"Step {step}/{n_steps}, Energy: {energies[-1]:.4f}, "
                      f"Acceptance: {n_accepted/n_total:.3f}")
        
        return {
            'configs': torch.cat(configs, dim=0),
            'energies': np.array(energies),
            'acceptance_rate': n_accepted / n_total
        }

# Example usage
def train_patch_van(patch_size=4, lattice_size=16, beta=0.44, n_train_steps=1000):
    """Train a VAN on patches extracted from Metropolis samples"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # First, generate training data using standard Metropolis
    print("Generating training patches...")
    train_patches = []
    
    # Simple Metropolis to generate patches
    config = torch.randint(0, 2, (1, 1, lattice_size, lattice_size)).float() * 2 - 1
    config = config.to(device)
    
    for _ in range(n_train_steps):
        # Random spin flip
        i, j = torch.randint(0, lattice_size, (2,))
        config_new = config.clone()
        config_new[0, 0, i, j] *= -1
        
        # Metropolis acceptance
        delta_E = 2 * config[0, 0, i, j] * (
            config[0, 0, (i+1)%lattice_size, j] + 
            config[0, 0, (i-1)%lattice_size, j] +
            config[0, 0, i, (j+1)%lattice_size] + 
            config[0, 0, i, (j-1)%lattice_size]
        )
        
        if torch.rand(1) < torch.exp(-beta * delta_E):
            config = config_new
            
        # Extract random patch for training
        if _ % 10 == 0:  # Every 10 steps
            pi = torch.randint(patch_size//2, lattice_size - patch_size//2, (1,)).item()
            pj = torch.randint(patch_size//2, lattice_size - patch_size//2, (1,)).item()
            patch = config[:, :, pi-patch_size//2:pi+patch_size//2, 
                                pj-patch_size//2:pj+patch_size//2]
            train_patches.append(patch.cpu())
    
    train_patches = torch.cat(train_patches, dim=0)
    print(f"Generated {len(train_patches)} training patches")
    
    # Initialize patch VAN
    patch_van = PatchVAN(
        patch_size=patch_size,
        depth=3,
        width=16,  # Smaller network for patches
        kernel_rad=1,
        device=device
    )
    
    # Training would go here - using your existing training code
    # For now, just return untrained model
    return patch_van

# Run example
if __name__ == "__main__":
    # Train patch VAN
    patch_van = train_patch_van(patch_size=2, lattice_size=16, beta=0.44)
    
    # Run enhanced Metropolis
    sampler = EnhancedMetropolis(
        patch_van=patch_van,
        lattice_size=16,
        beta=0.44,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    results = sampler.run(n_steps=1000, n_patches_per_step=10)
    
    print(f"\nFinal acceptance rate: {results['acceptance_rate']:.3f}")
    print(f"Final energy: {results['energies'][-1]:.4f}")
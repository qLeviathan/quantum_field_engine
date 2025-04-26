
import torch

# --- Phase Angle Rebalancer ---
def rebalance_phase_angles(field, alpha=0.1):
    '''
    Smooth the phase angles (θ) across the field to prevent clustering collapse.
    
    Args:
        field (dict): QFNN field containing 'θ' (angles) and 'ψ' (wavefunction)
        alpha (float): smoothing strength (0=none, 1=full rebalance)
    
    Returns:
        Updated field dictionary with rebalanced θ and ψ
    '''
    θ = field["θ"]

    # Target: evenly spaced phase steps
    N = len(θ)
    ideal_θ = torch.linspace(-torch.pi, torch.pi, N, device=θ.device)

    # Smoothly interpolate between current θ and ideal θ
    new_θ = (1 - alpha) * θ + alpha * ideal_θ

    # Update wavefunction ψ to match new angles
    r = torch.abs(field["ψ"])
    real_part = r * torch.cos(new_θ)
    imag_part = r * torch.sin(new_θ)
    ψ_components = torch.stack([real_part, imag_part], dim=1)
    ψ_rebalanced = torch.view_as_complex(ψ_components)

    # Update field
    field["θ"] = new_θ
    field["ψ"] = ψ_rebalanced

    return field

# --- Example Usage ---
if __name__ == "__main__":
    # Dummy test field
    θ = torch.rand(10) * 2 * torch.pi - torch.pi
    r = torch.ones(10)
    real = r * torch.cos(θ)
    imag = r * torch.sin(θ)
    ψ = torch.view_as_complex(torch.stack([real, imag], dim=1))

    field = {"θ": θ, "ψ": ψ}
    field = rebalance_phase_angles(field)

    print(f"✅ Field rebalanced. New θ: {field['θ']}")

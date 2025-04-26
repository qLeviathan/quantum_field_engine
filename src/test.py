import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Setup simplified AxiomaticQuantumField parts for the test
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Φ = (1 + 5**0.5) / 2  # Golden ratio

class SimpleField:
    def __init__(self):
        self.config = {
            "η_0": 0.002,
            "phase_weight": 0.75,
            "radius_weight": 0.2,
            "step_size": 0.01,
            "steps": 100,
            "entropy_weight": 0.01,
            "epsilon": 1e-8,
        }
        self.device = DEVICE
        self.golden_ratio = Φ

    def init_field(self, tokens):
        N = len(tokens)
        i = torch.arange(N, dtype=torch.float32, device=self.device)
        θ = (i * self.golden_ratio * 2 * torch.pi) % (2 * torch.pi)
        r = 0.3 + 0.7 * torch.linspace(0, 1, N, device=self.device)
        real_part = r * torch.cos(θ)
        imag_part = r * torch.sin(θ)
        ψ_components = torch.stack([real_part, imag_part], dim=1)
        ψ = torch.view_as_complex(ψ_components)
        return ψ, θ

# Tokenize simple text manually
tokens = ["the", "cat", "sat", "on", "the", "mat"]
model = SimpleField()
ψ, θ = model.init_field(tokens)
θ_target = torch.roll(θ, shifts=-1)

# Storage
phase_drift = []
r_energy = []
coherence_curve = []
eta_curve = []
beta_curve = []

for step in range(100):
    β = 0.1 + 20.0 * (step / 100)**2
    inverse_beta_noise = 1.0 / β
    beta_curve.append(β)
    
    noise_injection = inverse_beta_noise * torch.randn_like(ψ)
    ψ = ψ + noise_injection
    ψ = F.normalize(ψ, p=2, dim=0)

    θ_pred = torch.angle(ψ)
    θ_error = torch.abs(θ_pred - θ_target)
    phase_loss = torch.mean(θ_error ** 2)

    η = (model.config["η_0"] +
         model.config["phase_weight"] * phase_loss +
         0.05 * (1.0 / β))
    eta_curve.append(η.item())

    coherence = torch.abs(torch.mean(torch.exp(1j * θ_pred)))
    phase_drift.append(phase_loss.item())
    r_energy.append(torch.mean(torch.abs(ψ)).item())
    coherence_curve.append(coherence.item())

# Plot results
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.plot(phase_drift)
plt.title("Phase Drift over Time")
plt.xlabel("Steps")
plt.ylabel("Phase Loss")

plt.subplot(2,2,2)
plt.plot(r_energy)
plt.title("Field Energy (r) over Time")
plt.xlabel("Steps")
plt.ylabel("Mean Radius")

plt.subplot(2,2,3)
plt.plot(coherence_curve)
plt.title("Coherence over Time")
plt.xlabel("Steps")
plt.ylabel("Coherence")

plt.subplot(2,2,4)
plt.plot(beta_curve, label='β (Entropy Shock)')
plt.plot(eta_curve, label='η (Adaptive Step Size)')
plt.title("β vs η Over Time")
plt.xlabel("Steps")
plt.ylabel("Value")
plt.legend()

plt.tight_layout()
plt.show()

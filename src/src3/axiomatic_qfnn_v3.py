import torch
import torch.nn.functional as F
import math

Φ = (1 + math.sqrt(5)) / 2  # Golden ratio
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "η_0": 0.002,
    "phase_weight": 0.75,
    "radius_weight": 0.2,
    "hebb_lr": 0.05,
    "hebb_decay": 0.99,
    "entropy_weight": 0.01,
    "diffusion_coef": 0.1,
    "step_size": 0.01,
    "steps": 500,
    "epsilon": 1e-8,
}

class AxiomaticQuantumFieldV3:
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.device = DEVICE
        self.golden_ratio = Φ

    def init_field(self, tokens):
        N = len(tokens)
        i = torch.arange(N, dtype=torch.float32, device=self.device)
        θ = (i * self.golden_ratio * 2 * torch.pi) % (2 * torch.pi)
        r = 0.3 + 0.7 * torch.sqrt((i + 1.0) / N)
        real_part = r * torch.cos(θ)
        imag_part = r * torch.sin(θ)
        ψ_components = torch.stack([real_part, imag_part], dim=1)
        ψ = torch.view_as_complex(ψ_components)
        W = torch.zeros((N, N), dtype=torch.cfloat, device=self.device)
        F_stack = torch.zeros_like(ψ)
        return {"θ": θ, "r": r, "ψ": ψ, "W": W, "F_stack": F_stack}

    def compute_loss(self, ψ, θ_target):
        θ_pred = torch.angle(ψ)
        phase_diff = (θ_pred - θ_target) % (2 * torch.pi)
        log_phase_error = torch.log(phase_diff + self.config["epsilon"])
        mag_sq = torch.abs(ψ) ** 2
        entropy = -torch.sum(mag_sq * torch.log(mag_sq + self.config["epsilon"]))
        return log_phase_error.mean(), entropy

    def heun_step(self, field, θ_target, step):
        ψ, W, F_stack = field["ψ"], field["W"], field["F_stack"]
        θ_pred = torch.angle(ψ)
        phase_diff = (θ_pred - θ_target) % (2 * torch.pi)
        phase_alignment = 1.0 - torch.mean(torch.abs(phase_diff)) / torch.pi

        adaptive_diffusion_coef = self.config["diffusion_coef"] * (2.0 - phase_alignment)
        diffusion_noise = torch.sqrt(2.0 * adaptive_diffusion_coef * self.config["step_size"]) * torch.randn_like(ψ)

        V = (W @ ψ).real + 0.95 * F_stack.real
        k1 = -1j * (-ψ + V * ψ)
        h = self.config["step_size"]
        ψ_mid = ψ + h * (k1 + diffusion_noise)
        ψ_mid = F.normalize(ψ_mid, p=2, dim=0)

        k2 = -1j * (-ψ_mid + V * ψ_mid)
        ψ_next = ψ + 0.5 * h * (k1 + k2) + diffusion_noise
        ψ_next = F.normalize(ψ_next, p=2, dim=0)

        ΔW = self.config["hebb_lr"] * torch.outer(ψ_next.conj(), ψ_next)
        W_next = self.config["hebb_decay"] * W + ΔW
        F_stack_next = 0.95 * F_stack + ψ_next

        return {
            "θ": torch.angle(ψ_next),
            "r": torch.abs(ψ_next),
            "ψ": ψ_next,
            "W": W_next,
            "F_stack": F_stack_next
        }

    def vectorized_field_update(self, field, θ_target):
        ψ, W, F_stack = field["ψ"], field["W"], field["F_stack"]

        V = (W @ ψ).real + 0.95 * F_stack.real
        H_ψ = -ψ + V * ψ

        adaptive_diffusion_coef = torch.tensor(self.config["diffusion_coef"], device=self.device)

        diffusion_noise = torch.sqrt(2.0 * adaptive_diffusion_coef * self.config["step_size"]) * torch.randn_like(ψ)

        ψ_next = ψ + self.config["step_size"] * (-1j * H_ψ + diffusion_noise)
        ψ_next = F.normalize(ψ_next, p=2, dim=0)

        ΔW = self.config["hebb_lr"] * torch.outer(ψ_next.conj(), ψ_next)
        W_next = self.config["hebb_decay"] * W + ΔW
        F_stack_next = 0.95 * F_stack + ψ_next

        return {
            "θ": torch.angle(ψ_next),
            "r": torch.abs(ψ_next),
            "ψ": ψ_next,
            "W": W_next,
            "F_stack": F_stack_next
        }
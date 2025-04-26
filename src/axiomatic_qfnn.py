import torch
import math
import numpy as np
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

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

class AxiomaticQuantumField:
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.device = DEVICE
        self.golden_ratio = Φ

    
    def adaptive_entropy_boost(self, phase_loss, base_weight=0.01, alpha=3.0):
        """
        Boost entropy weight proportionally to phase misalignment (phase loss).
        """
        return base_weight * (1.0 + alpha * phase_loss.item())

    def tokenize(self, text):
        return word_tokenize(text.lower())

    def init_field(self, tokens):
        N = len(tokens)
        i = torch.arange(N, dtype=torch.float32, device=self.device)
        θ = (i * self.golden_ratio * 2 * torch.pi) % (2 * torch.pi)
        r = 0.3 + 0.7 * torch.linspace(0, 1, N, device=self.device)
        real_part = r * torch.cos(θ)
        imag_part = r * torch.sin(θ)
        ψ_components = torch.stack([real_part, imag_part], dim=1)
        ψ = torch.view_as_complex(ψ_components)
        W = torch.zeros((N, N), dtype=torch.cfloat, device=self.device)
        F_stack = torch.zeros_like(ψ)
        return {"θ": θ, "r": r, "ψ": ψ, "W": W, "F_stack": F_stack}

    def compute_loss(self, ψ, θ_target):
        θ_pred = torch.angle(ψ)
        phase_loss = torch.mean((θ_pred - θ_target) ** 2)
        mag_sq = torch.abs(ψ) ** 2
        entropy = -torch.sum(mag_sq * torch.log(mag_sq + self.config["epsilon"]))
        return phase_loss, entropy

    def heun_step(self, field, θ_target, step):
        ψ, W, F_stack = field["ψ"], field["W"], field["F_stack"]
        θ_pred = torch.angle(ψ)
        θ_error = torch.abs(θ_pred - θ_target)
        phase_loss = torch.mean(θ_error ** 2)
        mag_sq = torch.abs(ψ) ** 2
        entropy = -torch.sum(mag_sq * torch.log(mag_sq + self.config["epsilon"]))
        entropy_weight_now = self.adaptive_entropy_boost(phase_loss, base_weight=self.config["entropy_weight"])
        loss_total = phase_loss + entropy_weight_now * entropy

        # --- Simulated Annealing Scheduler ---
        if step == 0:
            self.prev_field = {"ψ": ψ.clone(), "W": W.clone(), "F_stack": F_stack.clone()}
            self.prev_loss_total = loss_total
        else:
            ΔE = loss_total - self.prev_loss_total
            T = 1.0 * (1.0 - step / self.config["steps"])
            adaptive_T = T * (1.0 + 3.0 * phase_loss.item())  # loss-controlled diffusion
            if ΔE <= 0:
                accept = True
            else:
                p_accept = torch.exp(-ΔE / (adaptive_T + self.config["epsilon"]))
                accept = torch.rand(1, device=self.device) < p_accept
            if accept:
                self.prev_field = {"ψ": ψ.clone(), "W": W.clone(), "F_stack": F_stack.clone()}
                self.prev_loss_total = loss_total
            else:
                # Rollback
                ψ = self.prev_field["ψ"].clone()
                W = self.prev_field["W"].clone()
                F_stack = self.prev_field["F_stack"].clone()
                loss_total = self.prev_loss_total


        N = len(ψ)
        k = max(1, int(0.25 * N))
        topk = torch.topk(θ_error, k=k)
        mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        mask[topk.indices] = True

        ψ_selected = ψ.clone()
        ψ_selected[~mask] = ψ.detach()[~mask]

        memory_pressure = torch.sum(torch.abs(W), dim=1)
        η = (self.config["η_0"] + 
             self.config["phase_weight"] * phase_loss + 
             self.config["radius_weight"] * memory_pressure)

        V = (W @ ψ).real + 0.95 * F_stack.real
        k1 = -1j * (-ψ + V * ψ)
        h = self.config["step_size"] * η.unsqueeze(1)
        ψ_mid = ψ_selected + h.squeeze() * k1
        ψ_mid = F.normalize(ψ_mid, p=2, dim=0)
        k2 = -1j * (-ψ_mid + V * ψ_mid)
        ψ_next = ψ_selected + 0.5 * h.squeeze() * (k1 + k2)
        ψ_next = F.normalize(ψ_next, p=2, dim=0)

        ΔW = self.config["hebb_lr"] * torch.outer(ψ_next.conj(), ψ_next)
        W_next = self.config["hebb_decay"] * W + ΔW
        F_stack_next = 0.95 * F_stack + ψ_next

        coherence = torch.abs(torch.mean(torch.exp(1j * torch.angle(ψ_next))))
        if coherence > 0.99:
            β = 0.1 + 20.0 * (step / self.config["steps"])**2
            noise_scale = 1.0 / β
            ψ_next = ψ_next + noise_scale * torch.randn_like(ψ_next)
            ψ_next = F.normalize(ψ_next, p=2, dim=0)
            print(f"⚡ Entropy shock at step {step} | β={β:.2f}")

        r_next = torch.abs(ψ_next)
        θ_next = torch.angle(ψ_next)

        return {"θ": θ_next, "r": r_next, "ψ": ψ_next, "W": W_next, "F_stack": F_stack_next}, loss_total, phase_loss, entropy, coherence

    def evolve(self, text):
        tokens = self.tokenize(text)
        if len(tokens) < 3:
            print(f"⚠️ Skipping short input ({len(tokens)} tokens): {repr(text[:50])}...")
            return None, {k: [] for k in ["loss", "phase_loss", "entropy", "coherence"]}
        field = self.init_field(tokens)
        θ_target = torch.roll(field["θ"], shifts=-1)
        history = {"loss": [], "phase_loss": [], "entropy": [], "coherence": []}
        for step in range(self.config["steps"]):
            field, loss, phase_loss, entropy, coherence = self.heun_step(field, θ_target, step)
            history["loss"].append(loss.item())
            history["phase_loss"].append(phase_loss.item())
            history["entropy"].append(entropy.item())
            history["coherence"].append(coherence.item())
            if step % 50 == 0 or step == self.config["steps"] - 1:
                print(f"[{step}] Loss: {loss:.4f} | Phase: {phase_loss:.4f} | Entropy: {entropy:.4f} | Coherence: {coherence:.4f}")
        return field, history
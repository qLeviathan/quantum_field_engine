
import torch
import torch.nn.functional as F
from axiomatic_qfnn_v3 import AxiomaticQuantumFieldV3

class QFNNFieldExpanderV3:
    def __init__(self, config=None):
        self.qfnn = AxiomaticQuantumFieldV3(config)
        self.device = self.qfnn.device

    def generate_sequence(self, prompt_tokens, num_new_tokens=10, temperature=1.0):
        # Initialize field from prompt
        field = self.qfnn.init_field(prompt_tokens)
        sequence = prompt_tokens.copy()

        for _ in range(num_new_tokens):
            # Short field evolution to allow ψ stabilization
            for _ in range(30):
                field = self.qfnn.vectorized_field_update(field, torch.roll(field["θ"], shifts=-1))

            # Sample next token phase using path integral logic
            θ_candidates = torch.linspace(-torch.pi, torch.pi, 128, device=self.device)
            log_probs = []

            for θ_candidate in θ_candidates:
                θ_new = torch.cat([field["θ"], θ_candidate.unsqueeze(0)])
                r_new = torch.cat([field["r"], torch.tensor([1.0], device=self.device)])

                real_part = r_new * torch.cos(θ_new)
                imag_part = r_new * torch.sin(θ_new)
                ψ_components = torch.stack([real_part, imag_part], dim=1)
                ψ_new = torch.view_as_complex(ψ_components)

                mag_sq = torch.abs(ψ_new) ** 2
                entropy = -torch.sum(mag_sq * torch.log(mag_sq + self.qfnn.config["epsilon"]))

                phase_diff = (θ_new[:-1] - θ_new[1:]) % (2 * torch.pi)
                log_phase_error = torch.log(phase_diff + self.qfnn.config["epsilon"])

                action = log_phase_error.mean() + self.qfnn.config["entropy_weight"] * entropy
                log_probs.append(-action / temperature)

            log_probs = torch.stack(log_probs)
            probs = F.softmax(log_probs, dim=0)
            idx = torch.multinomial(probs, 1).item()
            next_θ = θ_candidates[idx]

            # Expand θ and r
            field["θ"] = torch.cat([field["θ"], next_θ.unsqueeze(0)])
            field["r"] = torch.cat([field["r"], torch.tensor([1.0], device=self.device)])

            # Expand ψ
            real_part = field["r"] * torch.cos(field["θ"])
            imag_part = field["r"] * torch.sin(field["θ"])
            ψ_components = torch.stack([real_part, imag_part], dim=1)
            field["ψ"] = torch.view_as_complex(ψ_components)

            # Expand W (memory matrix)
            old_N = field["W"].shape[0]
            new_N = old_N + 1
            W_new = torch.zeros((new_N, new_N), dtype=torch.cfloat, device=self.device)
            W_new[:old_N, :old_N] = field["W"]
            field["W"] = W_new

            # Expand F_stack
            field["F_stack"] = torch.cat([field["F_stack"], torch.zeros(1, dtype=torch.cfloat, device=self.device)])

            # Add dummy token to sequence (for now, phase-based decoding later)
            sequence.append(f"θ={next_θ.item():.2f}")

        return sequence

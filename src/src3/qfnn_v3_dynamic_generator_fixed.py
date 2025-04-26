import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from axiomatic_qfnn_v3 import AxiomaticQuantumFieldV3

class QFNNFieldExpanderV3:
    def __init__(self, model_name="gpt2", config=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.qfnn = AxiomaticQuantumFieldV3(config)

        # Build Polar Bank dynamically
        self.build_polar_bank()

    def build_polar_bank(self):
        """Create polar coordinates for the tokenizer vocabulary."""
        vocab_size = self.tokenizer.vocab_size
        self.polar_bank = []

        θs = (torch.arange(vocab_size, device=self.device) * (1.61803398875 * 2 * torch.pi)) % (2 * torch.pi)
        rs = 0.3 + 0.7 * torch.sqrt((torch.arange(vocab_size, device=self.device) + 1.0) / vocab_size)

        self.polar_bank = torch.stack([rs * torch.cos(θs), rs * torch.sin(θs)], dim=1)

    def decode_phase(self, r, θ):
        """Decode a phase (r, θ) into the closest real token."""
        query = torch.tensor([r * torch.cos(θ), r * torch.sin(θ)], device=self.device)
        dists = torch.norm(self.polar_bank - query.unsqueeze(0), dim=1)
        idx = torch.argmin(dists)
        token = self.tokenizer.decode([idx])
        return token

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

            # Decode real token from polar coordinates
            decoded_token = self.decode_phase(1.0, next_θ)
            sequence.append(decoded_token)

        return sequence


import torch
import numpy as np
from qfnn_token_fieldbank import build_token_fieldbank
from axiomatic_qfnn_annealed import AxiomaticQuantumField
from nltk.tokenize import word_tokenize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Expand QFNN field by generating new tokens with stability patches ---
class QFNNFieldExpanderStable:
    def __init__(self, model_name="gpt2", max_tokens=5000):
        self.token_bank, self.tokenizer = build_token_fieldbank(model_name, max_tokens)
        self.model = AxiomaticQuantumField()
        
        # Filter clean tokens: remove special GPT artifacts
        self.token_list = [tok for tok in self.token_bank.keys() if tok.isalpha() and len(tok) > 1]
        self.phase_vectors = torch.stack([self.token_bank[tok] for tok in self.token_list]).to(DEVICE)

    def expand_field(self, prompt, num_new_tokens=10, steps_per_evolve=500, diversity_penalty=0.05):
        tokens = self.model.tokenize(prompt)
        field = self.model.init_field(tokens)
        θ_target = torch.roll(field["θ"], shifts=-1)

        generated_tokens = tokens.copy()
        used_token_indices = set()

        for _ in range(num_new_tokens):
            for step in range(steps_per_evolve):
                field, loss, phase_loss, entropy, coherence = self.model.heun_step(field, θ_target, step)

            # Predict next token
            ψ_last = field["ψ"][-1].unsqueeze(0)  # Take last evolved token
            distances = torch.abs(ψ_last - self.phase_vectors)

            # Add diversity penalty for already used tokens
            distances = distances.squeeze()
            for idx in used_token_indices:
                distances[idx] += diversity_penalty

            best_idx = torch.argmin(distances)
            next_token = self.token_list[best_idx]
            used_token_indices.add(best_idx)

            # Append new token
            generated_tokens.append(next_token)

            # Expand field with new ψ
            new_ψ = self.phase_vectors[best_idx].unsqueeze(0)
            field["ψ"] = torch.cat([field["ψ"], new_ψ])
            field["W"] = torch.nn.functional.pad(field["W"], (0, 1, 0, 1))
            field["F_stack"] = torch.cat([field["F_stack"], new_ψ])

            # Update θ and r
            θ_new = torch.angle(new_ψ.squeeze())
            r_new = torch.abs(new_ψ.squeeze())
            field["θ"] = torch.cat([field["θ"], θ_new.unsqueeze(0)])
            field["r"] = torch.cat([field["r"], r_new.unsqueeze(0)])

            # Update θ_target for new expanded field
            θ_target = torch.roll(field["θ"], shifts=-1)

        return " ".join(generated_tokens)

# --- Example Usage ---
if __name__ == "__main__":
    expander = QFNNFieldExpanderStable()
    prompt = "the river flows through the dream"
    output = expander.expand_field(prompt, num_new_tokens=20)
    print(f"🌌 QFNN Completion: {output}")

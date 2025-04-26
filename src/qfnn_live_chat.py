
import torch
import numpy as np
from axiomatic_qfnn import AxiomaticQuantumField
from nltk.tokenize import word_tokenize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def symbolic_field_completion(prompt, model=None, max_tokens=20, steps=500):
    if model is None:
        model = AxiomaticQuantumField()

    tokens = model.tokenize(prompt)
    if len(tokens) < 2:
        return "⚠️ Prompt too short to evolve."

    field = model.init_field(tokens)
    θ_target = torch.roll(field["θ"], shifts=-1)

    for step in range(steps):
        field, loss, phase_loss, entropy, coherence = model.heun_step(field, θ_target, step)

    # Evolve field and decode next steps
    ψ = field["ψ"].detach().cpu()
    r = torch.abs(ψ).numpy()
    θ = torch.angle(ψ).numpy()

    # Sort tokens by ascending θ
    sorted_indices = np.argsort(θ)
    sorted_tokens = [tokens[i] for i in sorted_indices]

    generated = []
    for idx in range(min(max_tokens, len(sorted_tokens))):
        generated.append(sorted_tokens[idx])

    return " ".join(generated)

# --- Live Chat Interface ---
if __name__ == "__main__":
    print("🌌 QFNN Live Chat (Symbolic Phase Inference Mode)")
    print("Type your prompt. Type 'exit' to quit.")
    model = AxiomaticQuantumField()

    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            break
        completion = symbolic_field_completion(prompt, model)
        print(f"QFNN: {completion}\n")

# quantum_field_engine_monolithic.py
import torch
import math
from nltk.tokenize import word_tokenize
from text_decoder import field_to_sentence

import nltk
nltk.download("punkt")

# --- CONFIG ---
Φ = (1 + math.sqrt(5)) / 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG = {
    "η_0": 0.002,
    "λ": 0.75,           # phase loss to learning rate sensitivity
    "μ": 0.2,            # memory modulation coefficient
    "hebb_lr": 0.05,
    "hebb_decay": 0.99,
    "entropy_weight": 0.01,
    "pareto_k": 0.25,
    "h_base": 0.01,
    "steps": 500,
    "epsilon": 1e-8,
}

# --- TOKENIZER ---
def tokenize(text):
    return word_tokenize(text.lower())

# --- INIT FIELD ---
def init_field(tokens):
    N = len(tokens)
    i = torch.arange(N, dtype=torch.float32, device=DEVICE)
    θ = (i * Φ * 2 * torch.pi) % (2 * torch.pi)
    r = 0.3 + 0.7 * torch.linspace(0, 1, N, device=DEVICE)
    ψ = torch.randn(N, 2, device=DEVICE)
    ψ = torch.view_as_complex(ψ)
    W = torch.zeros((N, N), dtype=torch.cfloat, device=DEVICE)
    return θ, r, ψ, W

# --- LOSS + ENTROPY ---
def compute_loss(ψ, θ_pred, θ_target):
    phase_loss = torch.mean((θ_pred - θ_target) ** 2)
    mag_sq = torch.abs(ψ) ** 2
    entropy = -torch.sum(mag_sq * torch.log(mag_sq + CONFIG["epsilon"]))
    return phase_loss, entropy

# --- PARETO MASK ---
def pareto_mask(metric, k):
    threshold = torch.quantile(metric, 1 - k)
    return metric >= threshold

# --- HEBBIAN UPDATE ---
def update_hebbian(W, ψ, η, decay):
    ΔW = η * torch.outer(ψ.conj(), ψ)
    return decay * W + ΔW

# --- SCHRÖDINGER PROPAGATION ---
def propagate(ψ, V, h):
    k1 = -ψ + V * ψ
    ψ_tilde = ψ + h * k1
    k2 = -(ψ_tilde) + V * ψ_tilde
    return ψ + 0.5 * h * (k1 + k2)
    
# --- Shock with cohernece approaching .99---

def apply_inverse_beta_shock(ψ, step, T, beta_min=0.1, beta_max=20.0, α=2.0, δ=0.1, Φ=(1 + math.sqrt(5)) / 2):
    β = beta_min + (beta_max - beta_min) * (step / T)**α * (1 + δ * math.sin(2 * math.pi * Φ * step / T))
    noise = (1.0 / β) * torch.randn_like(ψ)
    return ψ + noise

# --- COUPLED SYSTEM EVOLUTION ---
def evolve(text):
    tokens = tokenize(text)
    N = len(tokens)
    θ, r, ψ, W = init_field(tokens)
    θ_target = torch.roll(θ, shifts=-1)

    for step in range(CONFIG["steps"]):
        # Phase
        θ_pred = torch.angle(ψ).contiguous()

        # Loss
        phase_loss, entropy = compute_loss(ψ, θ_pred, θ_target)
        loss_total = phase_loss + CONFIG["entropy_weight"] * entropy

        # Vectorized per-token learning rate (add memory pressure)
        memory_pressure = torch.abs(W).sum(dim=1)
        η = CONFIG["η_0"] + CONFIG["λ"] * phase_loss + CONFIG["μ"] * memory_pressure.mean()

        # Pareto excitation (phase error magnitude)
        excitation = torch.abs(θ_pred - θ_target)
        mask = pareto_mask(excitation, CONFIG["pareto_k"])
        ψ_selected = ψ.clone()
        ψ_selected[~mask] = ψ[~mask].detach()

        # Compute V(r, θ, t) from memory
        V = torch.abs(W @ ψ).real  # simplified potential from memory activation

        # Propagate
        ψ = propagate(ψ_selected, V, CONFIG["h_base"] * η)

        # Hebbian update
        W = update_hebbian(W, ψ, CONFIG["hebb_lr"], CONFIG["hebb_decay"])

        # Logging
        if step % 50 == 0 or step == CONFIG["steps"] - 1:
            coherence = torch.abs(torch.mean(torch.exp(1j * θ_pred)))
            if coherence > 0.99:
                ψ = apply_inverse_beta_shock(ψ, step, CONFIG["steps"])
                print(f"⚡ Entropy shock injected at step {step} | β adjusted for phase exploration.")

            print(f"[{step}] Loss: {loss_total:.4f} | Phase: {phase_loss:.4f} | Entropy: {entropy:.4f} | Coherence: {coherence:.4f}")



def compose_fields(fields, method='average'):
    assert len(fields) > 1, "Need at least two fields to compose."

    shapes = [f["ψ"].shape for f in fields]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError("All fields must have the same ψ dimensionality.")

    ψ_stack = torch.stack([f["ψ"].to(DEVICE) for f in fields])
    W_stack = torch.stack([f["W"].to(DEVICE) for f in fields])
    F_stack = torch.stack([f["F_stack"].to(DEVICE) for f in fields])

    if method == 'average':
        ψ_fused = ψ_stack.mean(dim=0)
        W_fused = W_stack.mean(dim=0)
        F_fused = F_stack.mean(dim=0)
    elif method == 'sum':
        ψ_fused = ψ_stack.sum(dim=0)
        W_fused = W_stack.sum(dim=0)
        F_fused = F_stack.sum(dim=0)
    elif method == 'max':
        ψ_fused = ψ_stack.max(dim=0).values
        W_fused = W_stack.max(dim=0).values
        F_fused = F_stack.max(dim=0).values
    else:
        raise ValueError("Unknown composition method.")

    return {
        "ψ": ψ_fused,
        "W": W_fused,
        "F_stack": F_fused,
    }
def run_qfnn_on_corpus(text, window_size=32, stride=16):
    from nltk.tokenize import sent_tokenize

    checkpoints = []
    sentences = sent_tokenize(text)
    current = ""
    
    for sentence in sentences:
        if len(current.split()) < window_size:
            current += " " + sentence
        else:
            evolve_from_text(current.strip())
            checkpoint = torch.load("semantic_field_checkpoint.pt")
            checkpoints.append(checkpoint)
            current = sentence
    
    if current.strip():
        evolve_from_text(current.strip())
        checkpoint = torch.load("semantic_field_checkpoint.pt")
        checkpoints.append(checkpoint)
    
    return checkpoints


'''
# EXAMPLE: Compose 2 different field runs
evolve_from_text("a field of intelligence evolves")
cp1 = torch.load("semantic_field_checkpoint.pt")

evolve_from_text("memory and resonance organize coherence")
cp2 = torch.load("semantic_field_checkpoint.pt")

fused = compose_fields([cp1, cp2])
print("🧠 Fused Field Decoded:", field_to_sentence(fused["decoded"]))

# EXAMPLE: Run across a large text
long_text = open("some_text_file.txt").read()
checkpoints = run_qfnn_on_corpus(long_text)


'''

# --- RUN ---
if __name__ == "__main__":
    text = "consciousness emerges from coherent interference patterns in the semantic quantum field"
    evolve(text)

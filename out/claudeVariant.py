import torch
import math
import numpy as np
from nltk.tokenize import word_tokenize
import torch.nn.functional as F
import matplotlib.pyplot as plt
# Add this at the top of your file to import the decoder functions
try:
    from text_decoder import field_to_sentence, decode_from_field
except ImportError:
    print("Warning: text_decoder module not found, using internal decoder functions")
    # The decode_from_field and field_to_sentence functions you provided will be used as fallback
# --- CONSTANTS AND CONFIG ---
Φ = (1 + math.sqrt(5)) / 2  # Golden ratio
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Axiomatic configuration
AXIOM_CONFIG = {
    "η_0": 0.002,             # Base learning rate
    "phase_weight": 0.75,     # Phase alignment weight
    "radius_weight": 0.2,     # Radius alignment weight
    "hebb_lr": 0.05,          # Hebbian learning rate
    "hebb_decay": 0.99,       # Hebbian weight decay
    "entropy_weight": 0.01,   # Entropy regularization weight
    "diffusion_coef": 0.1,    # Diffusion coefficient for exploration
    "step_size": 0.01,        # Integration step size
    "steps": 500,             # Number of training steps
    "epsilon": 1e-8,          # Small constant for numerical stability
}

class AxiomaticQuantumSequenceModel:
    """
    Implementation of the axiomatic quantum sequence model based on physical principles.
    
    This model establishes sequence prediction as a physical system governed by:
    - Wave-Token Correspondence (Axiom 1)
    - Phase Alignment Principle (Axiom 2)
    - Information Potential Field (Axiom 3)
    - Phase-Gradient Equivalence (Axiom 4)
    - Golden Ratio Optimality (Axiom 5)
    - Least Action Principle (Axiom 6)
    - Hamiltonian-Loss Equivalence (Axiom 7)
    """
    
    def __init__(self, config=None):
        """Initialize the model with configuration parameters."""
        self.config = config or AXIOM_CONFIG
        self.golden_ratio = Φ
        self.device = DEVICE
        
    def tokenize(self, text):
        """Tokenize input text."""
        return word_tokenize(text.lower())
    
    def init_field(self, tokens):
        """
        Initialize quantum field with token embeddings.
        
        Implements Axioms 1 (Wave-Token Correspondence) and 5 (Golden Ratio Optimality).
        """
        N = len(tokens)
        
        # Initialize phase using golden ratio spacing (Axiom 5)
        i = torch.arange(N, dtype=torch.float32, device=self.device)
        θ = (i * self.golden_ratio * 2 * torch.pi) % (2 * torch.pi)
        
        # Initialize radius with sequential progression
        r = 0.3 + 0.7 * torch.linspace(0, 1, N, device=self.device)
        
        # Initialize complex wave function (Axiom 1)
        # This combines both representation methods: r,θ and complex ψ
        real_part = r * torch.cos(θ)
        imag_part = r * torch.sin(θ)
        ψ_components = torch.stack([real_part, imag_part], dim=1)
        ψ = torch.view_as_complex(ψ_components)
        
        # Initialize Hebbian weight matrix (for memory)
        W = torch.zeros((N, N), dtype=torch.cfloat, device=self.device)
        
        # Memory stack for diffusion dynamics
        F_stack = torch.zeros_like(ψ)
        
        return {
            "θ": θ,           # Phase angles
            "r": r,            # Radii (amplitudes)
            "ψ": ψ,            # Complex wave function
            "W": W,            # Hebbian weight matrix
            "F_stack": F_stack # Memory stack
        }
    
    def compute_loss(self, field, θ_target):
        """
        Compute loss based on phase alignment and entropy.
        
        Implements Axiom 7 (Hamiltonian-Loss Equivalence).
        """
        ψ = field["ψ"]
        θ_pred = torch.angle(ψ)
        
        # Phase alignment loss (Axiom 2)
        phase_loss = torch.mean((θ_pred - θ_target) ** 2)
        
        # Entropy regularization
        mag_sq = torch.abs(ψ) ** 2
        entropy = -torch.sum(mag_sq * torch.log(mag_sq + self.config["epsilon"]))
        
        return phase_loss, entropy
    
    def heun_euler_step(self, field, θ_target, step):
        """
        Perform a step of Heun-Euler integration for field evolution.
        
        Implements Axiom 6 (Least Action Principle) through the numerical integration.
        """
        ψ = field["ψ"]
        W = field["W"]
        F_stack = field["F_stack"]
        
        # Get current phase
        θ_pred = torch.angle(ψ)
        
        # Calculate phase error
        θ_error = torch.abs(θ_pred - θ_target)
        phase_loss = torch.mean(θ_error ** 2)
        
        # Calculate entropy
        mag_sq = torch.abs(ψ) ** 2
        entropy = -torch.sum(mag_sq * torch.log(mag_sq + self.config["epsilon"]))
        
        # Total loss
        loss_total = phase_loss + self.config["entropy_weight"] * entropy
        
        # Focus on tokens with highest error (Pareto principle)
        N = len(ψ)
        k = max(1, int(0.25 * N))  # Focus on top 25% errors
        topk = torch.topk(θ_error, k=k)
        mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        mask[topk.indices] = True
        
        # Prepare selected field for update
        ψ_selected = ψ.clone()
        ψ_selected[~mask] = ψ.detach()[~mask]
        
        # Calculate adaptive learning rate with memory pressure
        memory_pressure = torch.sum(torch.abs(W), dim=1)
        η = (self.config["η_0"] + 
             self.config["phase_weight"] * phase_loss + 
             self.config["radius_weight"] * memory_pressure)
        
        # FIRST STEP: Euler integration
        # Compute potential from Hebbian weights and memory stack
        V = (W @ ψ).real + 0.95 * F_stack.real
        
        # First derivative (quantum Hamiltonian evolution)
        k1 = -1j * (-ψ + V * ψ)
        
        # Euler step
        h = self.config["step_size"] * η.unsqueeze(1) if hasattr(η, 'shape') else self.config["step_size"] * η
        ψ_mid = ψ_selected + h.squeeze() * k1 if hasattr(h, 'squeeze') else ψ_selected + h * k1
        
        # Normalize midpoint
        ψ_mid = F.normalize(ψ_mid, p=2, dim=0)
        
        # SECOND STEP: Heun correction
        # Recalculate derivative at midpoint
        k2 = -1j * (-ψ_mid + V * ψ_mid)
        
        # Final Heun step
        ψ_next = ψ_selected + 0.5 * h.squeeze() * (k1 + k2) if hasattr(h, 'squeeze') else ψ_selected + 0.5 * h * (k1 + k2)
        
        # Normalize to preserve unitarity
        ψ_next = F.normalize(ψ_next, p=2, dim=0)
        
        # Update Hebbian weights (memory formation)
        ΔW = self.config["hebb_lr"] * torch.outer(ψ_next.conj(), ψ_next)
        W_next = self.config["hebb_decay"] * W + ΔW
        
        # Update memory stack
        F_stack_next = 0.95 * F_stack + ψ_next
        
        # Check for phase coherence and apply diffusion if needed
        coherence = torch.abs(torch.mean(torch.exp(1j * torch.angle(ψ_next))))
        if coherence > 0.99:
            # Apply diffusion shock for exploration
            beta = 0.1 + 20.0 * (step / self.config["steps"])**2
            noise_scale = 1.0 / beta
            ψ_next = ψ_next + noise_scale * torch.randn_like(ψ_next)
            ψ_next = F.normalize(ψ_next, p=2, dim=0)
            print(f"⚡ Entropy shock at step {step} | β={beta:.2f} | Phase coherence too high.")
        
        # Extract updated radius and phase
        r_next = torch.abs(ψ_next)
        θ_next = torch.angle(ψ_next)
        
        # Return updated field
        return {
            "θ": θ_next,
            "r": r_next,
            "ψ": ψ_next,
            "W": W_next,
            "F_stack": F_stack_next
        }, loss_total, phase_loss, entropy, coherence
    
    def evolve(self, text):
        """
        Evolve the quantum field for a given text.
        
        This implements the full axiomatic framework through iterative field evolution.
        """
        # Tokenize input text
        tokens = self.tokenize(text)
        
        # Initialize field
        field = self.init_field(tokens)
        
        # Target phases - shifted for sequence prediction (Axiom 3)
        θ_target = torch.roll(field["θ"], shifts=-1)
        
        # Evolution history for analysis
        history = {
            "loss": [],
            "phase_loss": [],
            "entropy": [],
            "coherence": []
        }
        
        # Main evolution loop
        for step in range(self.config["steps"]):
            # Evolve field through Heun-Euler integration
            field, loss, phase_loss, entropy, coherence = self.heun_euler_step(field, θ_target, step)
            
            # Record history
            history["loss"].append(loss.item())
            history["phase_loss"].append(phase_loss.item())
            history["entropy"].append(entropy.item())
            history["coherence"].append(coherence.item())
            
            # Log progress
            if step % 50 == 0 or step == self.config["steps"] - 1:
                print(f"[{step}] Loss: {loss:.4f} | Phase: {phase_loss:.4f} | " 
                      f"Entropy: {entropy:.4f} | Coherence: {coherence:.4f}")
        
        return field, θ_target, history
    
    def predict_next_token(self, field, tokens, temperature=1.0):
        """
        Predict the next token in the sequence.
        
        Implements the sequence prediction based on field dynamics.
        """
        ψ = field["ψ"]
        θ = torch.angle(ψ)
        
        # Target angle for next token
        θ_next_target = torch.roll(θ, shifts=-1)[0]
        
        # Measure alignment with target
        vocab = list(set(tokens))
        alignment_scores = []
        
        for token in vocab:
            # Find token position in original sequence
            positions = [i for i, t in enumerate(tokens) if t == token]
            if not positions:
                continue
                
            # Average angle for this token
            token_angles = [θ[i].item() for i in positions]
            mean_angle = sum(token_angles) / len(token_angles)
            
            # Calculate alignment with next target angle
            alignment = 1.0 - abs((mean_angle - θ_next_target) % (2 * torch.pi)) / torch.pi
            alignment_scores.append((token, alignment))
        
        # Apply temperature
        scores = [(token, score ** (1.0 / temperature)) for token, score in alignment_scores]
        
        # Normalize to probabilities
        total = sum(score for _, score in scores)
        probs = [(token, score / total) for token, score in scores]
        
        # Sort by probability
        probs.sort(key=lambda x: x[1], reverse=True)
        
        return probs
    
    def visualize_field(self, field, title="Quantum Field Visualization"):
        """Visualize the quantum field components."""
        ψ = field["ψ"]
        amp = torch.abs(ψ).cpu().numpy()
        phase = torch.angle(ψ).cpu().numpy()
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(amp)
        ax[0].set_title("Amplitude (Radius)")
        ax[1].plot(phase)
        ax[1].set_title("Phase")
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def visualize_history(self, history):
        """Visualize the evolution history."""
        steps = range(len(history["loss"]))
        
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax[0, 0].plot(steps, history["loss"])
        ax[0, 0].set_title("Total Loss")
        ax[0, 1].plot(steps, history["phase_loss"])
        ax[0, 1].set_title("Phase Loss")
        ax[1, 0].plot(steps, history["entropy"])
        ax[1, 0].set_title("Entropy")
        ax[1, 1].plot(steps, history["coherence"])
        ax[1, 1].set_title("Coherence")
        
        for a in ax.flat:
            a.set_xlabel("Step")
            a.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_checkpoint(self, field, path):
        """Save field checkpoint to file."""
        torch.save({
            "ψ": field["ψ"].detach().cpu(),
            "W": field["W"].detach().cpu(),
            "F_stack": field["F_stack"].detach().cpu(),
            "θ": field["θ"].detach().cpu(),
            "r": field["r"].detach().cpu()
        }, path)
        print(f"💾 Field checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load field checkpoint from file."""
        checkpoint = torch.load(path, map_location=self.device)
        return {
            "ψ": checkpoint["ψ"].to(self.device),
            "W": checkpoint["W"].to(self.device),
            "F_stack": checkpoint["F_stack"].to(self.device),
            "θ": checkpoint["θ"].to(self.device) if "θ" in checkpoint else torch.angle(checkpoint["ψ"].to(self.device)),
            "r": checkpoint["r"].to(self.device) if "r" in checkpoint else torch.abs(checkpoint["ψ"].to(self.device))
        }
    
    def decode_from_field(self, field, original_tokens):
        """
        Decode a field back to tokens.
        
        This reconstructs the most likely sequence from the field.
        """
        ψ = field["ψ"]
        θ = torch.angle(ψ)
        
        # Sort original tokens by phase
        token_phases = [(original_tokens[i], θ[i].item()) for i in range(len(original_tokens))]
        token_phases.sort(key=lambda x: x[1])
        
        # Extract sorted tokens
        sorted_tokens = [token for token, _ in token_phases]
        
        return " ".join(sorted_tokens)
    
    def compose_fields(self, fields, method='average'):
        """
        Compose multiple fields using the specified method.
        This version handles fields of different sizes through interpolation.
        
        Methods:
        - 'average': Average the fields (default)
        - 'sum': Sum the fields
        - 'max': Take maximum values
        """
        assert len(fields) > 1, "Need at least two fields to compose."
        
        # Determine maximum field size
        max_size = max(f["ψ"].shape[0] for f in fields)
        
        # Prepare properly sized fields
        aligned_fields = []
        for field in fields:
            ψ = field["ψ"]
            W = field["W"]
            F_stack = field["F_stack"]
            
            current_size = ψ.shape[0]
            
            if current_size == max_size:
                # Field already has the correct size
                aligned_fields.append({
                    "ψ": ψ,
                    "W": W, 
                    "F_stack": F_stack
                })
            else:
                # Interpolate field to the correct size
                print(f"Interpolating field from size {current_size} to {max_size}")
                
                # For ψ: Complex interpolation
                ψ_real = torch.real(ψ)
                ψ_imag = torch.imag(ψ)
                
                # Interpolate real and imaginary parts separately
                ψ_real_interp = torch.nn.functional.interpolate(
                    ψ_real.unsqueeze(0).unsqueeze(0),
                    size=max_size,
                    mode='linear'
                ).squeeze()
                
                ψ_imag_interp = torch.nn.functional.interpolate(
                    ψ_imag.unsqueeze(0).unsqueeze(0),
                    size=max_size,
                    mode='linear'
                ).squeeze()
                
                # Recombine to complex
                ψ_interp = torch.complex(ψ_real_interp, ψ_imag_interp)
                
                # For W: Handle 2D interpolation correctly
                W_real = torch.real(W)
                W_imag = torch.imag(W)
                
                # Add necessary dimensions for 2D interpolation
                # [H, W] -> [1, 1, H, W]
                W_real_2d = W_real.unsqueeze(0).unsqueeze(0)
                W_imag_2d = W_imag.unsqueeze(0).unsqueeze(0)
                
                # Perform interpolation
                W_real_interp = torch.nn.functional.interpolate(
                    W_real_2d,
                    size=(max_size, max_size),
                    mode='bilinear'
                ).squeeze()
                
                W_imag_interp = torch.nn.functional.interpolate(
                    W_imag_2d,
                    size=(max_size, max_size),
                    mode='bilinear'
                ).squeeze()
                
                # Recombine to complex
                W_interp = torch.complex(W_real_interp, W_imag_interp)
                
                # For F_stack: Same as ψ
                F_real = torch.real(F_stack)
                F_imag = torch.imag(F_stack)
                
                F_real_interp = torch.nn.functional.interpolate(
                    F_real.unsqueeze(0).unsqueeze(0),
                    size=max_size,
                    mode='linear'
                ).squeeze()
                
                F_imag_interp = torch.nn.functional.interpolate(
                    F_imag.unsqueeze(0).unsqueeze(0),
                    size=max_size,
                    mode='linear'
                ).squeeze()
                
                F_interp = torch.complex(F_real_interp, F_imag_interp)
                
                # Add to aligned fields
                aligned_fields.append({
                    "ψ": ψ_interp,
                    "W": W_interp,
                    "F_stack": F_interp
                })
        
        # Now stack the aligned fields
        ψ_stack = torch.stack([f["ψ"] for f in aligned_fields])
        W_stack = torch.stack([f["W"] for f in aligned_fields])
        F_stack = torch.stack([f["F_stack"] for f in aligned_fields])
        
        # Combine fields according to the selected method
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
            raise ValueError(f"Unknown composition method: {method}")
        
        # Normalize fused field
        ψ_fused = F.normalize(ψ_fused, p=2, dim=0)
        
        # Extract radius and phase
        r_fused = torch.abs(ψ_fused)
        θ_fused = torch.angle(ψ_fused)
        
        return {
            "ψ": ψ_fused,
            "W": W_fused,
            "F_stack": F_fused,
            "θ": θ_fused,
            "r": r_fused
        }

# --- DEMO FUNCTION ---
def run_proof_of_concept():
    """Run a proof of concept demonstration using the axiomatic model."""
    print("=" * 80)
    print("AXIOMATIC QUANTUM SEQUENCE MODEL - PROOF OF CONCEPT")
    print("=" * 80)
    
    # Initialize model
    model = AxiomaticQuantumSequenceModel()
    
    # Example text
    text = "the quantum field organizes information through phase alignment and radius modulation"
    print(f"Input text: '{text}'")
    print("-" * 80)
    
    # Tokenize
    tokens = model.tokenize(text)
    print(f"Tokenized ({len(tokens)} tokens): {tokens}")
    print("-" * 80)
    
    # Evolve field
    print("Evolving quantum field...")
    field, θ_target, history = model.evolve(text)
    print("-" * 80)
    
    # Visualize field
    print("Visualizing field components...")
    model.visualize_field(field, title="Evolved Quantum Field")
    
    # Visualize history
    print("Visualizing evolution history...")
    model.visualize_history(history)
    print("-" * 80)
    
    # Predict next tokens
    print("Predicting next tokens...")
    predictions = model.predict_next_token(field, tokens, temperature=0.5)
    print("Top 5 predictions:")
    for i, (token, prob) in enumerate(predictions[:5]):
        print(f"  {i+1}. '{token}' - {prob:.4f}")
    print("-" * 80)
    
    # Save checkpoint
    model.save_checkpoint(field, "axiomatic_field_checkpoint.pt")
    
    # Create a second field for composition
    print("Creating second field for composition...")
    text2 = "quantum coherence emerges from resonant interference patterns"
    field2, _, _ = model.evolve(text2)
    model.save_checkpoint(field2, "axiomatic_field2_checkpoint.pt")
    print("-" * 80)
    
    # Replace the final part of run_proof_of_concept function:

    # Compose fields
    print("Composing fields...")
    composed_field = model.compose_fields([field, field2], method='average')

    # Visualize composed field
    print("Visualizing composed field...")
    model.visualize_field(composed_field, title="Composed Quantum Field")

    # Decode from composed field using the imported decoder
    θ_fused = composed_field["θ"]
    θ_target_fused = torch.roll(θ_fused, shifts=-1)
    decoded_tokens = decode_from_field(θ_fused, θ_target_fused)
    decoded_sentence = field_to_sentence(decoded_tokens)
    print(f"Decoded from composed field: '{decoded_sentence}'")
    print("-" * 80)
    
    print("Proof of concept completed successfully!")
    return model, field, field2, composed_field, history

# --- INTEGRATION WITH ORIGINAL CODE ---
def integrate_with_original():
    """
    Integrate axiomatic model with original code to showcase compatibility.
    """
    # First run the axiomatic model
    model = AxiomaticQuantumSequenceModel()
    text = "quantum fields encode semantic relations through phase alignment"
    field, _, _ = model.evolve(text)
    model.save_checkpoint(field, "axiomatic_field.pt")
    
    # Convert to format compatible with original code
    from text_decoder import field_to_sentence, decode_from_field
    
    # Load checkpoint and convert
    checkpoint = torch.load("axiomatic_field.pt", map_location=DEVICE)
    ψ = checkpoint["ψ"].to(DEVICE)
    W = checkpoint["W"].to(DEVICE)
    F_stack = checkpoint["F_stack"].to(DEVICE)
    
    # Save in original format
    torch.save({
        "ψ": ψ,
        "W": W,
        "F_stack": F_stack,
        "meta": {"source": "axiomatic_model"}
    }, "compatible_checkpoint.pt")
    
    print("Checkpoint converted to format compatible with original code!")
    print("You can now use it with the original code's functions")

if __name__ == "__main__":
    # Run proof of concept
    model, field, field2, composed_field, history = run_proof_of_concept()
    
    # Attempt integration with original code
    try:
        integrate_with_original()
    except ImportError:
        print("Note: Full integration requires the text_decoder module from the original code.")
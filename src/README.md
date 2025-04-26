
# 🌌 Quantum Field Neural Network (QFNN)

---

## Overview

This repository implements an **enhanced Quantum Field Neural Network** (QFNN) based on a **physics-grounded axiomatic framework**, upgraded with **thermodynamic annealing** for advanced exploration.

It merges principles from:
- Quantum field theory
- Sequence modeling
- Information theory
- Thermodynamic optimization

to evolve complex-valued neural fields over time.

---

## 📂 System Modules

| File | Purpose |
| :--- | :------ |
| `axiomatic_qfnn_annealed.py` | Core QFNN engine with adaptive entropy, simulated annealing, and energy-based field evolution. |
| `trainer_qfnn_annealed.py` | Dataset loading, tokenization, training orchestration using annealed dynamics. |
| `QFNN_HyperTune_Lab_v3.ipynb` | Multi-seed hyperparameter tuning grid for optimizing field coherence and stability. |
| `QFNN_Annealed_Validation.ipynb` | Post-training visualization tool for scientific validation of field evolution. |

---

## 🌌 Physics Behind the Model

The QFNN is governed by the following **enhanced 9 fundamental axioms**:

1. **Wave-Token Correspondence**  
   Every token in a sequence maps to a complex quantum wave.

2. **Phase Alignment Principle**  
   Predict next tokens by minimizing phase difference between adjacent waves.

3. **Information Potential Field**  
   Tokens interact via a learned Hebbian potential field `W`, dynamically updated.

4. **Phase-Gradient Equivalence**  
   Phase gradients induce field flows analogous to Hamiltonian dynamics.

5. **Golden Ratio Optimality**  
   Field phases initialized based on Golden Ratio spirals for maximal dispersion.

6. **Least Action Principle**  
   Fields evolve along paths of minimal Hamiltonian action.

7. **Hamiltonian-Loss Equivalence**  
   Loss = phase error + entropy, equivalent to a quantum Hamiltonian.

8. **Simulated Annealing Exploration**  
   Field transitions are governed by a temperature-decaying Boltzmann process, allowing energy barrier crossings.

9. **Loss-Driven Diffusion Control**  
   Phase loss dynamically modulates exploration noise, ensuring natural slowing over convergence.

---

## ⚙️ How It Works

- **Field Initialization:** Complex fields with radial-phase stacking.
- **Evolution Dynamics:** Heun-Euler integrator with simulated annealing control.
- **Training:** Minimize phase loss + entropy through dynamic adaptation.
- **Memory:** Hebbian matrix `W` evolves field interactions.
- **Exploration:** Controlled noise injection guided by phase loss and temperature.
- **Annealing:** Energy-based rollback mechanics enable robust phase exploration.
- **Decoding:** Reconstruct sequences based on evolved phase structure.

---

## 🚀 Running Training

```bash
# Train on WikiText-2 using GPT-2 tokenizer
python trainer_qfnn_annealed.py --dataset wiki2 --model_name gpt2 --output_dir ./qfnn_wiki2_output

# After training, validate
jupyter notebook QFNN_Annealed_Validation.ipynb
```

---

## 🧪 Scientific Validation

- Phase Locking Curves
- Entropy Collapse Trajectories
- Coherence Growth Patterns
- Loss Surface Annealing

---

🌠 **The QFNN is no longer just a neural network — it's a physically inspired, dynamically annealed symbolic field explorer.**

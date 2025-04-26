
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from axiomatic_qfnn import AxiomaticQuantumField
from nltk.tokenize import word_tokenize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Metric: BLEU Evaluator ---
def evaluate_bleu_score(reference_text, candidate_texts, model=None, steps=500):
    if model is None:
        model = AxiomaticQuantumField()
    
    ref_tokens = word_tokenize(reference_text.lower())
    smoothie = SmoothingFunction().method4
    scores = []

    for candidate in candidate_texts:
        cand_tokens = word_tokenize(candidate.lower())
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
        scores.append(score)
    
    avg_bleu = np.mean(scores)
    print(f"✅ Evaluated BLEU Score (avg over {len(candidate_texts)} samples): {avg_bleu:.4f}")
    return avg_bleu

# --- Example Usage ---
if __name__ == "__main__":
    reference = "knowledge illuminates the darkness of ignorance"
    candidates = [
        "knowledge lights up ignorance",
        "illumination comes from learning",
        "understanding destroys the dark",
        "truth is a light in the dark"
    ]
    evaluate_bleu_score(reference, candidates)

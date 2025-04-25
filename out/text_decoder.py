import nltk
import random
import torch
import numpy as np
from nltk.corpus import wordnet as wn
from collections import defaultdict


import torch
import math

# Vocabulary and golden angle phase encoding
Φ = (1 + math.sqrt(5)) / 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB = [
    "intelligence", "resonance", "energy", "coherence", "field",
    "quantum", "semantic", "self", "organize", "emerges", "memory",
]

indices = torch.arange(len(VOCAB), dtype=torch.float32, device=DEVICE)
VOCAB_PHASES = (indices * Φ * 2 * torch.pi) % (2 * torch.pi)

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

# --- Semantic Field Configuration ---
SEMANTIC_DOMAINS = {
    "physics": ["field", "energy", "resonance", "quantum", "wave", "particle", "phase", "coherence", 
               "entropy", "stability", "flux", "attractor", "frequency", "interference"],
    "cognition": ["intelligence", "memory", "semantic", "cognition", "neural", "emergence", "pattern", 
                 "learning", "self", "organize", "adaptive", "perception", "attention", "consciousness"],
    "structure": ["system", "network", "lattice", "structure", "architecture", "pattern", "organization",
                 "hierarchy", "complexity", "dynamic", "equilibrium", "transformation", "symmetry"]
}

# --- Templates based on different semantic relationships ---
TEMPLATES = {
    # Field and Energy templates
    "field_energy": [
        "The {0} of {1} drives {2} through {3}.",
        "A {0} of {1} establishes {2} within the {3}.",
        "{0} within the {1} field creates {2} through {3}.",
        "The {0} field generates {1} that transforms {2} into {3}.",
        "{0} flows through {1} to create {2} patterns of {3}.",
    ],
    
    # Emergence and Causation templates
    "emergence": [
        "{0} emerges from the {1} of {2} when {3} is present.",
        "Through {0}, {1} emerges as {2} interacts with {3}.",
        "The {0} of {1} enables the emergence of {2} through {3}.",
        "{0} and {1} together give rise to emergent {2} within {3}.",
        "When {0} reaches critical {1}, {2} emerges from {3}.",
    ],
    
    # Transformation and State templates
    "transformation": [
        "When {0} aligns with {1}, {2} transforms into {3}.",
        "{0} transforms {1} through a process of {2} and {3}.",
        "The {0} state shifts toward {1} as {2} influences {3}.",
        "Through {0}, {1} evolves into {2} with increased {3}.",
        "{0} serves as a catalyst for {1}, converting {2} into {3}.",
    ],
    
    # Relationship and Connection templates
    "connection": [
        "{0} and {1} generate a field of {2} that sustains {3}.",
        "The connection between {0} and {1} manifests as {2} within {3}.",
        "{0} binds to {1}, creating a {2} that permeates {3}.",
        "A resonance between {0} and {1} establishes {2} across {3}.",
        "The interaction of {0} with {1} creates {2} across the {3} spectrum.",
    ],
    
    # Process and Method templates
    "process": [
        "{0} constructs {1} through {2} while maintaining {3}.",
        "Through a process of {0}, {1} develops {2} within {3}.",
        "{0} operates on {1} through {2}, resulting in {3}.",
        "The {0} process transforms {1} into {2} through successive {3}.",
        "{0} facilitates {1} by organizing {2} into coherent {3}.",
    ]
}

# --- Semantic Part-of-Speech Mappings ---
POS_PREFERENCES = {
    "field": ["NOUN", "ADJ"],
    "energy": ["NOUN", "ADJ"],
    "resonance": ["NOUN", "VERB"],
    "quantum": ["ADJ", "NOUN"],
    "memory": ["NOUN"],
    "intelligence": ["NOUN", "ADJ"],
    "semantic": ["ADJ", "NOUN"],
    "coherence": ["NOUN", "ADJ"],
    "entropy": ["NOUN"],
    "system": ["NOUN"],
    "emergence": ["NOUN", "VERB"],
    "phase": ["NOUN"],
    "pattern": ["NOUN"],
    "structure": ["NOUN", "VERB"],
    "dynamic": ["ADJ", "NOUN"],
    "organization": ["NOUN"],
    "complexity": ["NOUN"],
    "stability": ["NOUN", "ADJ"]
}

# NLTK POS to WordNet POS mapping
NLTK_TO_WN = {
    'JJ': wn.ADJ,
    'JJR': wn.ADJ,
    'JJS': wn.ADJ,
    'NN': wn.NOUN,
    'NNS': wn.NOUN,
    'NNP': wn.NOUN,
    'NNPS': wn.NOUN,
    'RB': wn.ADV,
    'RBR': wn.ADV,
    'RBS': wn.ADV,
    'VB': wn.VERB,
    'VBD': wn.VERB,
    'VBG': wn.VERB,
    'VBN': wn.VERB,
    'VBP': wn.VERB,
    'VBZ': wn.VERB
}

# --- Context-Aware Synonym Expansion ---
def get_synonym(word, context=None, pos=None):
    """Get a semantically appropriate synonym based on context and part of speech."""
    if random.random() > 0.4:  # 60% chance to keep original word
        return word
        
    # If POS is not provided, try to infer it
    if not pos:
        # Default to most common POS for this word
        if word in POS_PREFERENCES:
            pos_tag = POS_PREFERENCES[word][0]
        else:
            # Use NLTK to guess POS
            pos_tag = nltk.pos_tag([word])[0][1]
            
        # Convert NLTK POS tag to WordNet POS
        pos = NLTK_TO_WN.get(pos_tag, None)
    
    # Find synsets with the appropriate POS
    if pos:
        synsets = wn.synsets(word, pos=pos)
    else:
        synsets = wn.synsets(word)
        
    if not synsets:
        return word
        
    # Choose a synset, giving preference to more common meanings
    weights = [1.0/(i+1) for i in range(min(3, len(synsets)))]
    total_weight = sum(weights)
    normalized_weights = [w/total_weight for w in weights]
    
    chosen_synset = np.random.choice(synsets[:3], p=normalized_weights)
    
    # Get lemmas and filter by context if provided
    lemmas = chosen_synset.lemmas()
    if not lemmas:
        return word
        
    # If context is provided, try to find related lemmas
    if context:
        context_synsets = []
        for ctx_word in context:
            context_synsets.extend(wn.synsets(ctx_word))
            
        # Calculate semantic similarity with context
        if context_synsets:
            lemma_scores = []
            for lemma in lemmas:
                lemma_name = lemma.name().replace('_', ' ')
                if lemma_name == word:
                    continue
                    
                # Skip multi-word expressions if original was single word
                if ' ' in lemma_name and ' ' not in word:
                    continue
                    
                # Calculate average similarity with context
                lemma_synsets = wn.synsets(lemma_name)
                if lemma_synsets:
                    similarities = []
                    for l_syn in lemma_synsets[:1]:
                        for c_syn in context_synsets[:3]:
                            try:
                                sim = l_syn.path_similarity(c_syn)
                                if sim is not None:
                                    similarities.append(sim)
                            except:
                                pass
                                
                    if similarities:
                        avg_sim = sum(similarities) / len(similarities)
                        lemma_scores.append((lemma_name, avg_sim))
            
            # Choose synonym with probability proportional to similarity
            if lemma_scores:
                lemma_scores.sort(key=lambda x: x[1], reverse=True)
                top_candidates = lemma_scores[:3]
                words, scores = zip(*top_candidates)
                
                # Normalize scores
                total = sum(scores)
                if total > 0:
                    probs = [s/total for s in scores]
                    return np.random.choice(words, p=probs)
    
    # Fallback to random choice from lemmas
    return random.choice(lemmas).name().replace('_', ' ')

# --- Domain Detection ---
def detect_domains(tokens):
    """Detect which semantic domains are represented in the tokens."""
    domain_scores = defaultdict(int)
    
    for token in tokens:
        for domain, words in SEMANTIC_DOMAINS.items():
            if token in words:
                domain_scores[domain] += 1
                
    # Return domains with at least one match, sorted by score
    valid_domains = [d for d, s in domain_scores.items() if s > 0]
    return sorted(valid_domains, key=lambda d: domain_scores[d], reverse=True)

# --- Template Selection based on Semantic Content ---
def select_template(tokens, domains):
    """Select an appropriate template based on token content and domains."""
    # Count tokens by type
    has_field = any(t in ["field", "space", "domain", "realm"] for t in tokens)
    has_energy = any(t in ["energy", "force", "power", "potential"] for t in tokens)
    has_process = any(t in ["process", "method", "system", "organize", "transformation"] for t in tokens)
    has_emergence = any(t in ["emergence", "emerges", "arise", "form", "develop"] for t in tokens)
    
    # Choose template category
    if "physics" in domains and (has_field or has_energy):
        category = "field_energy"
    elif has_emergence or "emergence" in tokens:
        category = "emergence"
    elif has_process:
        category = "process"
    elif len(tokens) >= 4:  # If we have enough tokens for relationships
        if random.random() < 0.5:
            category = "connection"
        else:
            category = "transformation"
    else:
        # Choose random category as fallback
        category = random.choice(list(TEMPLATES.keys()))
    
    # Select template from category
    return random.choice(TEMPLATES[category])

# --- Grammar Correction ---
def apply_grammar_fixes(sentence):
    """Apply basic grammar fixes to the generated sentence."""
    # Handle article usage
    for word in ["a", "an"]:
        pattern = f"{word} "
        if pattern in sentence:
            parts = sentence.split(pattern)
            for i in range(1, len(parts)):
                if parts[i] and parts[i][0].lower() in "aeiou":
                    correct_article = "an" if word == "a" else "a"
                    parts[i-1] += f"{correct_article} "
                else:
                    parts[i-1] += f"{word} "
            sentence = "".join(parts)
    
    # Fix double spaces
    while "  " in sentence:
        sentence = sentence.replace("  ", " ")
        
    # Ensure proper sentence ending
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
        
    return sentence

# --- Main Semantic Field to Sentence Conversion ---
def field_to_sentence(tokens, use_synonyms=True, structure_hint=None):
    """
    Convert a list of semantic tokens into a coherent sentence.
    
    Args:
        tokens (list): List of semantic tokens from the quantum field
        use_synonyms (bool): Whether to use synonym expansion
        structure_hint (str): Optional hint for sentence structure
        
    Returns:
        str: Generated sentence
    """
    # Ensure we have tokens
    if not tokens or len(tokens) == 0:
        return "The semantic field is empty."
    
    # Handle single token
    if len(tokens) == 1:
        return f"The field resonates with {tokens[0]}."
    
    # Clean up tokens (remove duplicates while preserving order)
    seen = set()
    cleaned_tokens = [t for t in tokens if not (t in seen or seen.add(t))]
    
    # Get primary semantic domains
    domains = detect_domains(cleaned_tokens)
    
    # If using synonyms, expand vocabulary
    if use_synonyms:
        # Create a context from all tokens
        context = cleaned_tokens.copy()
        expanded_tokens = []
        
        for token in cleaned_tokens:
            synonym = get_synonym(token, context=context)
            expanded_tokens.append(synonym)
    else:
        expanded_tokens = cleaned_tokens.copy()
    
    # Choose a template based on token content and domains
    if structure_hint and structure_hint in TEMPLATES:
        # Use suggested structure
        template = random.choice(TEMPLATES[structure_hint])
    else:
        template = select_template(cleaned_tokens, domains)
    
    # Count required placeholders
    n_placeholders = template.count("{")
    
    # Ensure we have enough tokens
    if len(expanded_tokens) < n_placeholders:
        # Pad with domain-appropriate words
        if domains:
            primary_domain = domains[0]
            domain_words = SEMANTIC_DOMAINS[primary_domain]
            while len(expanded_tokens) < n_placeholders:
                expanded_tokens.append(random.choice(domain_words))
        else:
            # Generic padding
            padding = ["field", "energy", "pattern", "system"]
            expanded_tokens.extend(padding)
            
    # Clip to required size
    tokens_to_use = expanded_tokens[:n_placeholders]
    
    # Format the template
    try:
        sentence = template.format(*tokens_to_use)
        sentence = apply_grammar_fixes(sentence)
        return sentence.capitalize()
    except Exception as e:
        # Fallback for any formatting errors
        fallback = " ".join(expanded_tokens[:6])
        return fallback.capitalize() + "."
        
# --- Paragraph Generation ---
def field_to_paragraph(token_groups, coherence=0.7):
    """
    Generate a paragraph from multiple token groups.
    
    Args:
        token_groups (list): List of token lists, each representing a semantic unit
        coherence (float): How semantically coherent the paragraph should be (0-1)
        
    Returns:
        str: Generated paragraph
    """
    sentences = []
    
    # Track used structures to avoid repetition
    used_structures = set()
    available_structures = list(TEMPLATES.keys())
    
    for i, tokens in enumerate(token_groups):
        # For first sentence, use neutral structure
        if i == 0:
            structure = random.choice(["field_energy", "emergence"])
            used_structures.add(structure)
        else:
            # Use coherence to determine if we should connect to previous sentence
            if random.random() < coherence:
                # Choose connecting structure
                connecting_structures = [s for s in available_structures if s not in used_structures]
                if not connecting_structures:
                    connecting_structures = available_structures
                
                structure = random.choice(connecting_structures)
                used_structures.add(structure)
                
                # Add some tokens from previous group for coherence
                prev_tokens = token_groups[i-1]
                shared_tokens = [t for t in prev_tokens if t in tokens]
                
                if not shared_tokens and prev_tokens:
                    # If no overlap, add a bridge token
                    bridge_token = random.choice(prev_tokens)
                    tokens = tokens + [bridge_token]
            else:
                structure = None
        
        # Generate sentence
        sentence = field_to_sentence(tokens, structure_hint=structure)
        sentences.append(sentence)
    
    # Combine into paragraph
    return " ".join(sentences)

# --- Concept Mapping for Metaphors ---
def map_to_metaphor(tokens, target_domain):
    """Map quantum/physics tokens to another conceptual domain."""
    # Define domain mappings (physics concepts → target domain concepts)
    domain_maps = {
        "economics": {
            "field": "market",
            "energy": "capital",
            "resonance": "trend",
            "quantum": "micro-economic",
            "wave": "cycle",
            "coherence": "coordination",
            "entropy": "volatility",
            "phase": "period",
            "emergence": "growth",
            "attractor": "equilibrium",
            "system": "economy",
            "pattern": "behavior"
        },
        "cognition": {
            "field": "mind",
            "energy": "attention",
            "resonance": "understanding",
            "quantum": "neuronal",
            "wave": "thought",
            "coherence": "clarity",
            "entropy": "confusion",
            "phase": "state",
            "emergence": "insight",
            "attractor": "habit",
            "system": "framework",
            "pattern": "memory"
        }
    }
    
    # Check if we have mappings for requested domain
    if target_domain not in domain_maps:
        return tokens
    
    mapping = domain_maps[target_domain]
    mapped_tokens = []
    
    for token in tokens:
        # Map token if it exists in mapping, otherwise keep original
        mapped_tokens.append(mapping.get(token, token))
        
    return mapped_tokens

# --- Example Usage ---
if __name__ == "__main__":
    # Test with sample decoded field tokens
    decoded_tokens = [
        "resonance", "field", "coherence", "memory", 
        "intelligence", "semantic", "energy"
    ]
    
    print("🧠 Decoded Tokens:", decoded_tokens)
    
    # Generate single sentence
    sentence = field_to_sentence(decoded_tokens)
    print("\n🗣️ Generated Sentence:")
    print(sentence)
    
    # Generate with synonym expansion
    sentence_with_synonyms = field_to_sentence(decoded_tokens, use_synonyms=True)
    print("\n🗣️ Generated Sentence (with synonyms):")
    print(sentence_with_synonyms)
    
    # Generate paragraph from multiple token groups
    token_groups = [
        ["field", "energy", "resonance", "coherence"],
        ["memory", "pattern", "semantic", "organization"],
        ["intelligence", "emergence", "complexity", "system"]
    ]
    
    paragraph = field_to_paragraph(token_groups)
    print("\n📝 Generated Paragraph:")
    print(paragraph)
    
    # Map to economics domain
    econ_tokens = map_to_metaphor(decoded_tokens, "economics")
    econ_sentence = field_to_sentence(econ_tokens)
    print("\n💹 Economic Metaphor:")
    print(econ_sentence)


def decode_from_field(θ_pred, θ_target=None, top_k=1):
    """
    Vectorized decoder: match θ_pred (or θ_target) to nearest VOCAB_PHASES.
    """
    if θ_target is not None:
        targets = θ_target[:-1]  # shape: (N-1,)
    else:
        targets = θ_pred[:-1]    # shape: (N-1,)

    # Expand target angles and VOCAB_PHASES to (N-1, V)
    θ_expanded = targets.unsqueeze(1)                 # (N-1, 1)
    phase_diff = torch.abs(θ_expanded - VOCAB_PHASES)  # (N-1, V)

    # Get indices of minimum distance
    indices = torch.argmin(phase_diff, dim=1)  # (N-1,)

    # Convert to tokens
    return [VOCAB[i] for i in indices.tolist()]

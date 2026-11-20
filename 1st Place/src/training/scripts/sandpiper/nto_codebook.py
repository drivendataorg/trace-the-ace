"""NTO Tutor Move Taxonomy: Table 1 of Zhou, Vanacore, Thompson, St John & Kizilcec
(National Tutoring Observatory / Cornell), arXiv:2603.05778.

The taxonomy is not included due to licensing. Populate NTO_TAXONOMY using the structure below.
See src/training/docs/sandpiper_codebooks.md or the paper (arXiv:2603.05778, Table 1) for all
32 definitions. For step 02b of the training pipeline.
"""

# category -> [(CODE, definition), ...]
# Example of one category; populate the rest using the same format from:
#   - Paper Table 1: https://arxiv.org/abs/2603.05778
#   - Sandpiper: src/training/docs/sandpiper_codebooks.md
NTO_TAXONOMY = {
    "TUTORING_SUPPORT": [
        ("ASKING_TO_CLARIFY_CONTEXT",
         "Tutor asks the student about the problem to understand the context."),
        # Add remaining TUTORING_SUPPORT codes here (5 total)
    ],
    # Add LEARNING_SUPPORT, SOCIAL_EMOTIONAL_SUPPORT, LOGISTICAL_SUPPORT below
}

ALL_CODES = [c for cat in NTO_TAXONOMY.values() for c, _ in cat] if NTO_TAXONOMY else []
LEARNING_SPECTRUM = [c for c, _ in NTO_TAXONOMY.get("LEARNING_SUPPORT", [])]
CONTROL_RANK = {c: i for i, c in enumerate(LEARNING_SPECTRUM)}


def codebook_text() -> str:
    """Render the codebook as prompt text.

    Returns:
        One ``[CATEGORY]`` header per category followed by ``- CODE: definition`` lines, with
        the names and definitions verbatim.
    """
    if not NTO_TAXONOMY or not NTO_TAXONOMY.get("LEARNING_SUPPORT"):
        raise ValueError(
            "NTO_TAXONOMY not populated. See docstring or "
            "src/training/docs/sandpiper_codebooks.md for definitions."
        )
    
    lines = []
    for cat, rows in NTO_TAXONOMY.items():
        lines.append(f"[{cat.replace('_', ' ')}]")
        for code, definition in rows:
            lines.append(f"- {code}: {definition}")
    return "\n".join(lines)

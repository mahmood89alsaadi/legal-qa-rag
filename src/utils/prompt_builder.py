"""
Prompt Builder — supports baseline, adversarial, and auditor-style prompts.
Based on prompt engineering study findings from Year 1 experiments.
"""

from typing import List, Optional


SYSTEM_PROMPTS = {
    "baseline": (
        "You are a legal expert. Answer the following multiple-choice question "
        "by selecting the most accurate option based on the provided regulatory context."
    ),
    "adversarial": (
        "You are a critical legal reviewer. Several answer options are deliberately misleading. "
        "Carefully analyze each option and the regulatory context before selecting the correct answer."
    ),
    "auditor": (
        "You are a regulatory compliance auditor. Your role is to interpret regulatory obligations "
        "with precision. Use only the provided legal context to select the most legally accurate answer."
    ),
}


def format_options(options: List[str]) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    return "\n".join(f"{label}. {text}" for label, text in zip(labels, options))


def build_prompt(
    question: str,
    options: List[str],
    context: Optional[str] = None,
    style: str = "baseline",
) -> str:
    """
    Build a full prompt for legal MCQ answering.

    Args:
        question: The legal question
        options: List of answer choices
        context: Retrieved legal passages (optional for no-RAG baseline)
        style: Prompt style — 'baseline', 'adversarial', or 'auditor'

    Returns:
        Full formatted prompt string
    """
    system = SYSTEM_PROMPTS.get(style, SYSTEM_PROMPTS["baseline"])
    formatted_options = format_options(options)

    context_block = ""
    if context:
        context_block = f"""
### Regulatory Context:
{context}
"""

    prompt = f"""{system}
{context_block}
### Question:
{question}

### Options:
{formatted_options}

### Instructions:
- Read the regulatory context carefully.
- Select the single best answer from the options above.
- Respond with ONLY the letter of your answer (e.g., A, B, C...).

### Answer:"""

    return prompt.strip()


def build_cot_prompt(
    question: str,
    options: List[str],
    context: Optional[str] = None,
) -> str:
    """
    Chain-of-thought prompt for generating reasoning traces (used in fine-tuning data generation).
    """
    formatted_options = format_options(options)

    context_block = f"\n### Regulatory Context:\n{context}\n" if context else ""

    prompt = f"""You are a legal reasoning expert. For the following legal question, 
think step by step before selecting your answer.
{context_block}
### Question:
{question}

### Options:
{formatted_options}

### Instructions:
1. Identify the key legal obligation or rule in the question.
2. Review each option carefully, noting why it may or may not apply.
3. Use the regulatory context to support your reasoning.
4. State your final answer as: "Answer: [LETTER]"

### Reasoning:"""

    return prompt.strip()

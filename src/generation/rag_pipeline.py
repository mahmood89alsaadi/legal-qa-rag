"""
RAG Pipeline for Legal MCQ Question Answering.
Retrieves relevant legal passages and injects them into LLM prompts.
"""

import json
from typing import List, Dict, Optional
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.prompt_builder import build_prompt


class RAGPipeline:
    """
    Full RAG pipeline:
    1. Retrieve top-K legal passages for each question
    2. Build context-aware prompt
    3. Run LLM inference
    4. Parse and return answer
    """

    def __init__(self, retriever: HybridRetriever, llm, config: Dict):
        self.retriever = retriever
        self.llm = llm
        self.top_k = config.get("top_k", 5)
        self.prompt_style = config.get("prompt_style", "baseline")
        self.mcq_size = config.get("mcq_size", 4)

    def answer(self, question: str, options: List[str]) -> Dict:
        # Step 1: Retrieve relevant context
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        context = "\n\n".join([r["text"] for r in retrieved])

        # Step 2: Build prompt
        prompt = build_prompt(
            question=question,
            options=options,
            context=context,
            style=self.prompt_style,
        )

        # Step 3: LLM inference
        raw_output = self.llm.generate(prompt)

        # Step 4: Parse answer
        answer = self._parse_answer(raw_output, options)

        return {
            "question": question,
            "predicted_answer": answer,
            "retrieved_passages": retrieved,
            "raw_output": raw_output,
            "prompt": prompt,
        }

    def _parse_answer(self, output: str, options: List[str]) -> str:
        """Extract the selected option label (A, B, C, ...) from model output."""
        output_upper = output.upper().strip()
        labels = [chr(65 + i) for i in range(len(options))]  # A, B, C, ...

        for label in labels:
            if output_upper.startswith(label):
                return label

        # Fallback: scan for first label mention
        for label in labels:
            if label in output_upper:
                return label

        return "UNKNOWN"

    def batch_answer(self, samples: List[Dict]) -> List[Dict]:
        """Run pipeline on a list of QA samples."""
        results = []
        for i, sample in enumerate(samples):
            print(f"[{i+1}/{len(samples)}] Processing: {sample['question'][:60]}...")
            result = self.answer(
                question=sample["question"],
                options=sample["options"],
            )
            result["gold_answer"] = sample.get("answer", None)
            results.append(result)
        return results

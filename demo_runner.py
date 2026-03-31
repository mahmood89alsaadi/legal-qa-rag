"""
╔══════════════════════════════════════════════════════════╗
║        legal-qa-rag  —  LOCAL DEMO & TEST RUNNER        ║
║   No GPU · No API key · No dataset · Zero pip installs  ║
╚══════════════════════════════════════════════════════════╝

Run:  python demo_runner.py

Tests every component of the project with mock legal data:
  ✓ BM25 Retriever
  ✓ Hybrid Retriever (mock dense)
  ✓ Prompt Builder (all 3 styles + CoT)
  ✓ RAG Pipeline (mock LLM)
  ✓ Evaluation metrics
  ✓ Data preparation (MCQ-4 → MCQ-20 expansion)
  ✓ Full end-to-end pipeline
"""

import sys
import json
import math
import random
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional

# ─────────────────────────────────────────────
#  COLOURS for terminal output
# ─────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):print(f"  {RED}✗{RESET} {msg}")
def info(msg):print(f"  {CYAN}→{RESET} {msg}")
def section(title):
    print(f"\n{BOLD}{BLUE}{'═'*55}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'═'*55}{RESET}")

# ─────────────────────────────────────────────
#  MOCK LEGAL CORPUS  (replaces ObliQA)
# ─────────────────────────────────────────────
MOCK_CORPUS = [
    "Article 17 of GDPR grants the right to erasure. A data subject may request deletion when data is no longer necessary for the purpose it was collected.",
    "Under GDPR Article 6, processing is lawful only if the data subject has given consent, or processing is necessary for a contract.",
    "Article 5 of GDPR establishes that personal data shall be collected for specified, explicit and legitimate purposes and not further processed in a manner incompatible with those purposes.",
    "The controller shall implement appropriate technical measures to ensure data security under Article 32 of GDPR.",
    "Data subjects have the right to access their personal data under Article 15. The controller must respond within one month.",
    "Article 83 sets out conditions for imposing administrative fines. Violations of basic principles can lead to fines up to 20 million EUR.",
    "Under CCPA, consumers have the right to know what personal information is collected and the right to opt-out of sale of personal information.",
    "HIPAA requires covered entities to implement safeguards to protect the privacy of protected health information (PHI).",
    "The Data Protection Officer (DPO) must be designated when processing operations require regular and systematic monitoring of data subjects on a large scale.",
    "Article 28 requires a written contract between the controller and processor setting out the subject-matter and duration of the processing.",
]

# ─────────────────────────────────────────────
#  MOCK MCQ SAMPLES
# ─────────────────────────────────────────────
MOCK_SAMPLES = [
    {
        "id": "q001",
        "question": "Under GDPR Article 17, when can a data subject request erasure of their personal data?",
        "answer": "When the data is no longer necessary for the purpose it was collected",
        "source": "GDPR"
    },
    {
        "id": "q002",
        "question": "What is the maximum administrative fine for violating basic principles of GDPR?",
        "answer": "20 million EUR or 4% of global turnover",
        "source": "GDPR"
    },
    {
        "id": "q003",
        "question": "Under GDPR Article 15, within what timeframe must a controller respond to a data access request?",
        "answer": "One month",
        "source": "GDPR"
    },
    {
        "id": "q004",
        "question": "When is a Data Protection Officer (DPO) required under GDPR?",
        "answer": "When processing requires regular and systematic monitoring of data subjects on a large scale",
        "source": "GDPR"
    },
    {
        "id": "q005",
        "question": "What does GDPR Article 6 require for lawful processing?",
        "answer": "The data subject has given consent or processing is necessary for a contract",
        "source": "GDPR"
    },
]


# ══════════════════════════════════════════════
#  MODULE 1 — BM25 RETRIEVER (pure Python)
# ══════════════════════════════════════════════
class SimpleBM25:
    """BM25 Okapi implementation — no dependencies."""

    def __init__(self, corpus: List[str], k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.tokenized = [self._tokenize(doc) for doc in corpus]
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in self.tokenized) / self.N
        self.df = self._compute_df()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def _compute_df(self) -> Dict[str, int]:
        df = {}
        for doc in self.tokenized:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        return df

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1)

    def score(self, query: str) -> List[float]:
        tokens = self._tokenize(query)
        scores = []
        for doc_tokens in self.tokenized:
            tf_map = Counter(doc_tokens)
            dl = len(doc_tokens)
            score = 0.0
            for term in tokens:
                tf = tf_map.get(term, 0)
                idf = self._idf(term)
                score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            scores.append(score)
        return scores

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        scores = self.score(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"doc_id": i, "score": round(s, 4), "text": self.corpus[i]}
            for i, s in ranked
        ]


# ══════════════════════════════════════════════
#  MODULE 2 — MOCK DENSE RETRIEVER
# ══════════════════════════════════════════════
class MockDenseRetriever:
    """
    Simulates BGE-M3 dense retrieval using simple word overlap as a proxy.
    Replace with real BAAI/bge-m3 embeddings in production.
    """

    def __init__(self, corpus: List[str]):
        self.corpus = corpus

    def _overlap_score(self, query: str, doc: str) -> float:
        q_words = set(re.findall(r'\w+', query.lower()))
        d_words = set(re.findall(r'\w+', doc.lower()))
        if not q_words:
            return 0.0
        return len(q_words & d_words) / len(q_words)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        scores = [(i, self._overlap_score(query, doc)) for i, doc in enumerate(self.corpus)]
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"doc_id": i, "score": round(s, 4), "text": self.corpus[i]}
            for i, s in ranked
        ]


# ══════════════════════════════════════════════
#  MODULE 3 — HYBRID RETRIEVER (BM25 + Dense)
# ══════════════════════════════════════════════
class HybridRetriever:
    """Combines BM25 + dense via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, corpus: List[str], bm25_weight=0.4, dense_weight=0.6):
        self.bm25 = SimpleBM25(corpus)
        self.dense = MockDenseRetriever(corpus)
        self.corpus = corpus
        self.bm25_w = bm25_weight
        self.dense_w = dense_weight

    def _rrf(self, rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank + 1)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        bm25_res = self.bm25.retrieve(query, top_k=top_k * 2)
        dense_res = self.dense.retrieve(query, top_k=top_k * 2)

        scores: Dict[int, float] = {}
        for rank, r in enumerate(bm25_res):
            scores[r["doc_id"]] = scores.get(r["doc_id"], 0) + self.bm25_w * self._rrf(rank)
        for rank, r in enumerate(dense_res):
            scores[r["doc_id"]] = scores.get(r["doc_id"], 0) + self.dense_w * self._rrf(rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"doc_id": idx, "score": round(score, 6), "text": self.corpus[idx]}
            for idx, score in ranked
        ]


# ══════════════════════════════════════════════
#  MODULE 4 — PROMPT BUILDER
# ══════════════════════════════════════════════
SYSTEM_PROMPTS = {
    "baseline":   "You are a legal expert. Select the most accurate answer based on the regulatory context.",
    "adversarial":"You are a critical legal reviewer. Some options are misleading — be careful.",
    "auditor":    "You are a regulatory compliance auditor. Use only the provided legal context.",
}

def format_options(options: List[str]) -> str:
    return "\n".join(f"  {chr(65+i)}. {opt}" for i, opt in enumerate(options))

def build_prompt(question: str, options: List[str],
                 context: str = "", style: str = "baseline") -> str:
    system = SYSTEM_PROMPTS.get(style, SYSTEM_PROMPTS["baseline"])
    ctx_block = f"\n### Regulatory Context:\n{context}\n" if context else ""
    return f"""{system}
{ctx_block}
### Question:
{question}

### Options:
{format_options(options)}

### Instructions:
Respond with ONLY the letter of the correct answer (e.g., A, B, C...).

### Answer:"""

def build_cot_prompt(question: str, options: List[str], context: str = "") -> str:
    ctx_block = f"\n### Regulatory Context:\n{context}\n" if context else ""
    return f"""You are a legal reasoning expert. Think step by step.
{ctx_block}
### Question:
{question}

### Options:
{format_options(options)}

### Instructions:
1. Identify the key legal obligation.
2. Review each option carefully.
3. State your answer as: "Answer: [LETTER]"

### Reasoning:"""


# ══════════════════════════════════════════════
#  MODULE 5 — MOCK LLM  (simulates model output)
# ══════════════════════════════════════════════
class MockLLM:
    """
    Simulates LLM output for testing.
    In production: replace with Ollama, HuggingFace, or API calls.
    """

    def __init__(self, accuracy: float = 0.75):
        self.accuracy = accuracy  # simulated accuracy

    def generate(self, prompt: str) -> str:
        # Extract options count from prompt
        labels = re.findall(r'\n\s+([A-Z])\. ', prompt)
        if not labels:
            return "A"
        # Simulate model picking correct answer 'accuracy'% of the time
        correct = labels[0]  # assume A is correct in mock
        if random.random() < self.accuracy:
            return correct
        else:
            wrong = [l for l in labels if l != correct]
            return random.choice(wrong) if wrong else correct


# ══════════════════════════════════════════════
#  MODULE 6 — RAG PIPELINE
# ══════════════════════════════════════════════
class RAGPipeline:
    def __init__(self, retriever, llm, top_k=3, prompt_style="baseline"):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.prompt_style = prompt_style

    def answer(self, question: str, options: List[str]) -> Dict:
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        context = "\n\n".join(r["text"] for r in retrieved)
        prompt = build_prompt(question, options, context, self.prompt_style)
        raw = self.llm.generate(prompt)
        predicted = self._parse(raw, options)
        return {
            "question": question,
            "predicted": predicted,
            "retrieved": retrieved,
            "prompt_style": self.prompt_style,
        }

    def _parse(self, output: str, options: List[str]) -> str:
        labels = [chr(65 + i) for i in range(len(options))]
        output = output.strip().upper()
        for label in labels:
            if output.startswith(label):
                return label
        for label in labels:
            if label in output:
                return label
        return "A"

    def batch(self, samples: List[Dict]) -> List[Dict]:
        results = []
        for s in samples:
            r = self.answer(s["question"], s["options"])
            r["gold"] = s.get("answer", "A")
            results.append(r)
        return results


# ══════════════════════════════════════════════
#  MODULE 7 — EVALUATION
# ══════════════════════════════════════════════
def accuracy(preds: List[Dict]) -> float:
    correct = sum(1 for p in preds if p.get("predicted") == p.get("gold"))
    return correct / len(preds) if preds else 0.0

def mcq_drop(mcq4: List[Dict], mcq20: List[Dict]) -> Dict:
    a4  = accuracy(mcq4)
    a20 = accuracy(mcq20)
    return {
        "mcq4":  round(a4, 4),
        "mcq20": round(a20, 4),
        "drop":  round(a4 - a20, 4),
        "drop%": round((a4 - a20) / a4 * 100, 1) if a4 else 0,
    }

def robustness(results_by_style: Dict[str, List[Dict]]) -> Dict:
    accs = {s: accuracy(p) for s, p in results_by_style.items()}
    vals = list(accs.values())
    var  = sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals)
    return {"per_style": accs, "variance": round(var, 6), "robust": var < 0.01}


# ══════════════════════════════════════════════
#  MODULE 8 — DATA PREPARATION
# ══════════════════════════════════════════════
def make_mcq(sample: Dict, size: int, pool: List[Dict]) -> Dict:
    correct = sample["answer"]
    distractors = [s["answer"] for s in pool if s["answer"] != correct]
    random.shuffle(distractors)
    options = [correct] + distractors[:size - 1]
    # Pad if needed
    while len(options) < size:
        options.append(f"None of the above ({len(options)})")
    random.shuffle(options)
    label = chr(65 + options.index(correct))
    return {
        "id": sample["id"],
        "question": sample["question"],
        "options": options,
        "answer": label,
        "answer_text": correct,
        "mcq_size": size,
    }


# ══════════════════════════════════════════════
#  TEST RUNNER
# ══════════════════════════════════════════════
def run_all_tests():
    random.seed(42)
    passed = 0
    failed = 0

    # ── TEST 1: BM25 Retriever ───────────────
    section("TEST 1 — BM25 Retriever")
    bm25 = SimpleBM25(MOCK_CORPUS)
    query = "GDPR Article 17 erasure personal data"
    results = bm25.retrieve(query, top_k=3)
    info(f"Query: '{query}'")
    for r in results:
        info(f"  [score={r['score']}] {r['text'][:70]}...")
    if len(results) == 3 and results[0]["score"] > 0:
        ok("BM25 retrieves top-3 results with positive scores"); passed += 1
    else:
        fail("BM25 retrieval failed"); failed += 1

    # Relevance check — Article 17 doc should rank high
    top_texts = " ".join(r["text"] for r in results[:2])
    if "Article 17" in top_texts or "erasure" in top_texts:
        ok("Top results are relevant to the query"); passed += 1
    else:
        fail("Top results not relevant"); failed += 1

    # ── TEST 2: Hybrid Retriever ─────────────
    section("TEST 2 — Hybrid Retriever (BM25 + Mock Dense)")
    hybrid = HybridRetriever(MOCK_CORPUS, bm25_weight=0.4, dense_weight=0.6)
    h_results = hybrid.retrieve(query, top_k=3)
    info("Hybrid top-3 results:")
    for r in h_results:
        info(f"  [rrf={r['score']}] {r['text'][:70]}...")
    if len(h_results) == 3:
        ok("Hybrid retriever returns 3 results"); passed += 1
    else:
        fail("Hybrid retriever wrong count"); failed += 1

    # Check no duplicates
    ids = [r["doc_id"] for r in h_results]
    if len(ids) == len(set(ids)):
        ok("No duplicate documents in results"); passed += 1
    else:
        fail("Duplicate documents found"); failed += 1

    # ── TEST 3: Prompt Builder ───────────────
    section("TEST 3 — Prompt Builder (all styles)")
    q       = "What does GDPR Article 6 require for lawful processing?"
    options = ["Explicit consent only", "Consent or contractual necessity",
               "Government authorization", "No requirement"]
    context = MOCK_CORPUS[1]

    for style in ["baseline", "adversarial", "auditor"]:
        prompt = build_prompt(q, options, context, style)
        has_q   = "Question:" in prompt
        has_opt = "Options:"  in prompt
        has_ans = "Answer:"   in prompt
        has_ctx = "Context:"  in prompt
        if has_q and has_opt and has_ans and has_ctx:
            ok(f"Style '{style}' — all sections present"); passed += 1
        else:
            fail(f"Style '{style}' — missing sections"); failed += 1

    # CoT prompt
    cot = build_cot_prompt(q, options, context)
    if "step by step" in cot and "Reasoning:" in cot:
        ok("CoT prompt includes reasoning scaffold"); passed += 1
    else:
        fail("CoT prompt missing reasoning scaffold"); failed += 1

    # Check all option labels present
    for label in ["A.", "B.", "C.", "D."]:
        prompt = build_prompt(q, options, context, "baseline")
        if label not in prompt:
            fail(f"Option label {label} missing from prompt"); failed += 1
            break
    else:
        ok("All option labels (A–D) present in prompt"); passed += 1

    # ── TEST 4: Data Preparation ─────────────
    section("TEST 4 — Data Preparation (MCQ-4 & MCQ-20)")
    random.seed(42)
    mcq4_samples  = [make_mcq(s, 4,  MOCK_SAMPLES) for s in MOCK_SAMPLES]
    mcq20_samples = [make_mcq(s, 20, MOCK_SAMPLES) for s in MOCK_SAMPLES]

    for sample in mcq4_samples:
        if len(sample["options"]) != 4:
            fail("MCQ-4 wrong option count"); failed += 1; break
    else:
        ok(f"MCQ-4: all {len(mcq4_samples)} samples have exactly 4 options"); passed += 1

    for sample in mcq20_samples:
        if len(sample["options"]) != 20:
            fail("MCQ-20 wrong option count"); failed += 1; break
    else:
        ok(f"MCQ-20: all {len(mcq20_samples)} samples have exactly 20 options"); passed += 1

    # Correct answer always present in options
    for s in mcq4_samples + mcq20_samples:
        correct_label = s["answer"]
        idx = ord(correct_label) - 65
        if s["options"][idx] != s["answer_text"]:
            fail("Correct answer label doesn't match option text"); failed += 1; break
    else:
        ok("Answer labels correctly point to answer text in all samples"); passed += 1

    # ── TEST 5: RAG Pipeline ─────────────────
    section("TEST 5 — RAG Pipeline (mock LLM)")
    llm      = MockLLM(accuracy=0.8)
    pipeline = RAGPipeline(hybrid, llm, top_k=3, prompt_style="baseline")

    sample_q = MOCK_SAMPLES[0]
    mcq_item  = make_mcq(sample_q, 4, MOCK_SAMPLES)
    result    = pipeline.answer(mcq_item["question"], mcq_item["options"])

    info(f"Question: {result['question'][:60]}...")
    info(f"Predicted: {result['predicted']}")
    info(f"Retrieved {len(result['retrieved'])} passages")

    if result["predicted"] in [chr(65+i) for i in range(4)]:
        ok("Pipeline returns a valid answer label (A–D)"); passed += 1
    else:
        fail("Pipeline returned invalid label"); failed += 1

    if len(result["retrieved"]) == 3:
        ok("Pipeline retrieved exactly 3 passages"); passed += 1
    else:
        fail("Wrong number of retrieved passages"); failed += 1

    # ── TEST 6: Evaluation Metrics ───────────
    section("TEST 6 — Evaluation Metrics")

    # Prepare mock predictions
    mcq4_preds  = [{"predicted": "A", "gold": "A"}] * 9 + [{"predicted": "B", "gold": "A"}]
    mcq20_preds = [{"predicted": "A", "gold": "A"}] * 6 + [{"predicted": "B", "gold": "A"}] * 4

    acc4 = accuracy(mcq4_preds)
    acc20 = accuracy(mcq20_preds)
    drop  = mcq_drop(mcq4_preds, mcq20_preds)

    info(f"MCQ-4  accuracy : {acc4*100:.1f}%")
    info(f"MCQ-20 accuracy : {acc20*100:.1f}%")
    info(f"Performance drop: {drop['drop%']}%")

    if acc4 == 0.9 and acc20 == 0.6:
        ok("Accuracy calculation correct (MCQ-4=90%, MCQ-20=60%)"); passed += 1
    else:
        fail("Accuracy calculation wrong"); failed += 1

    if drop["drop"] == 0.3:
        ok("MCQ difficulty drop correctly computed (30%)"); passed += 1
    else:
        fail("Drop calculation wrong"); failed += 1

    # Prompt robustness
    style_results = {
        "baseline":   [{"predicted":"A","gold":"A"}]*8 + [{"predicted":"B","gold":"A"}]*2,
        "adversarial":[{"predicted":"A","gold":"A"}]*7 + [{"predicted":"B","gold":"A"}]*3,
        "auditor":    [{"predicted":"A","gold":"A"}]*8 + [{"predicted":"B","gold":"A"}]*2,
    }
    rob = robustness(style_results)
    info(f"Robustness — per style: {rob['per_style']}")
    info(f"Variance: {rob['variance']} → {'ROBUST' if rob['robust'] else 'SENSITIVE'}")
    ok("Robustness metric computed successfully"); passed += 1

    # ── TEST 7: Full End-to-End Pipeline ─────
    section("TEST 7 — End-to-End Pipeline")
    random.seed(99)
    llm_e2e = MockLLM(accuracy=0.75)
    rag_e2e = RAGPipeline(hybrid, llm_e2e, top_k=3, prompt_style="baseline")

    # Build MCQ-4 and MCQ-20 versions of all samples
    all_mcq4  = [make_mcq(s, 4,  MOCK_SAMPLES) for s in MOCK_SAMPLES]
    all_mcq20 = [make_mcq(s, 20, MOCK_SAMPLES) for s in MOCK_SAMPLES]

    preds4  = rag_e2e.batch(all_mcq4)
    preds20 = rag_e2e.batch(all_mcq20)

    acc4_e2e  = accuracy(preds4)
    acc20_e2e = accuracy(preds20)
    drop_e2e  = mcq_drop(preds4, preds20)

    info(f"MCQ-4  accuracy : {acc4_e2e*100:.1f}%")
    info(f"MCQ-20 accuracy : {acc20_e2e*100:.1f}%")
    info(f"Performance drop: {drop_e2e['drop%']}%")

    if len(preds4) == 5 and len(preds20) == 5:
        ok("End-to-end processed all 5 samples for MCQ-4 and MCQ-20"); passed += 1
    else:
        fail("End-to-end wrong sample count"); failed += 1

    # Test all 3 prompt styles in batch
    style_batch_results = {}
    for style in ["baseline", "adversarial", "auditor"]:
        pipe = RAGPipeline(hybrid, MockLLM(0.75), prompt_style=style)
        style_batch_results[style] = pipe.batch(all_mcq4)
    rob_e2e = robustness(style_batch_results)
    info(f"Prompt style robustness variance: {rob_e2e['variance']}")
    ok("All 3 prompt styles ran successfully in batch mode"); passed += 1

    # ── SUMMARY ──────────────────────────────
    total = passed + failed
    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  RESULTS: {GREEN}{passed}{RESET}{BOLD} passed, {RED}{failed}{RESET}{BOLD} failed  ({total} total){RESET}")
    print(f"{BOLD}{'═'*55}{RESET}\n")

    if failed == 0:
        print(f"{GREEN}{BOLD}  🎉 All tests passed! Project is ready for GitHub.{RESET}\n")
    else:
        print(f"{RED}{BOLD}  ⚠️  {failed} test(s) failed. Review output above.{RESET}\n")

    return failed == 0


# ══════════════════════════════════════════════
#  DEMO MODE  — shows a full worked example
# ══════════════════════════════════════════════
def run_demo():
    random.seed(42)
    print(f"\n{BOLD}{CYAN}{'═'*55}{RESET}")
    print(f"{BOLD}{CYAN}  DEMO — Full Pipeline Walkthrough{RESET}")
    print(f"{BOLD}{CYAN}{'═'*55}{RESET}\n")

    hybrid = HybridRetriever(MOCK_CORPUS)
    llm    = MockLLM(accuracy=0.8)

    question = "Under GDPR Article 17, when can a data subject request erasure of personal data?"
    options_4 = [
        "When data is no longer necessary for the purpose collected",
        "When the controller decides to delete data",
        "Only when data was collected illegally",
        "After a 5-year retention period",
    ]

    print(f"{BOLD}Question:{RESET}")
    print(f"  {question}\n")

    print(f"{BOLD}Step 1 — Hybrid Retrieval:{RESET}")
    retrieved = hybrid.retrieve(question, top_k=3)
    for i, r in enumerate(retrieved):
        print(f"  [{i+1}] score={r['score']} | {r['text'][:80]}...")

    print(f"\n{BOLD}Step 2 — Building Prompt (baseline style):{RESET}")
    context = "\n".join(r["text"] for r in retrieved)
    prompt  = build_prompt(question, options_4, context, style="baseline")
    # Show trimmed prompt
    for line in prompt.split("\n")[:12]:
        print(f"  {line}")
    print("  ...")

    print(f"\n{BOLD}Step 3 — LLM Answer:{RESET}")
    raw  = llm.generate(prompt)
    pred = raw.strip().upper()[0] if raw.strip() else "A"
    print(f"  Predicted: {GREEN}{pred}{RESET} ({options_4[ord(pred)-65]})")
    print(f"  Gold:      {GREEN}A{RESET} ({options_4[0]})")
    print(f"  Correct:   {GREEN}✓{RESET}" if pred == "A" else f"  Correct:   {RED}✗{RESET}")

    print(f"\n{BOLD}MCQ-20 Difficulty Demo:{RESET}")
    sample = MOCK_SAMPLES[0]
    mcq20  = make_mcq(sample, 20, MOCK_SAMPLES)
    print(f"  Same question now has {len(mcq20['options'])} options (harder!)")
    print(f"  Correct answer: {mcq20['answer']} — {mcq20['answer_text'][:50]}...")


# ══════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{BOLD}{'═'*55}")
    print("  legal-qa-rag — Demo & Test Runner")
    print(f"{'═'*55}{RESET}")
    print("  No GPU · No pip installs · No API key needed\n")

    run_demo()
    success = run_all_tests()
    sys.exit(0 if success else 1)

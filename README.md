# ⚖️ Legal QA with RAG + LLMs

> **PhD Research Project** — Improving Legal Question Answering using Retrieval-Augmented Generation and Fine-tuned LLMs, with a focus on high-difficulty MCQ settings (MCQ-4 → MCQ-20).

---

## 📌 Overview

Legal question answering is challenging due to:
- Precise regulatory language requirements
- Need for factually grounded answers
- Hallucination tendencies in standard LLMs

This project addresses these challenges by combining:
- **Hybrid Retrieval** (BM25 + Dense Embeddings via BGE-M3)
- **RAG Pipeline** (retrieved context injected into LLM prompts)
- **Parameter-Efficient Fine-Tuning** (LoRA/QLoRA)

---

## 🗂️ Project Structure

```
legal-qa-rag/
├── configs/                  # YAML configs for models, retrieval, training
├── data/
│   ├── raw/                  # Original datasets (e.g., ObliQA)
│   └── processed/            # MCQ-4 and MCQ-20 formatted data
├── notebooks/                # Exploratory analysis & experiments
├── results/                  # Evaluation outputs and metrics
├── scripts/                  # CLI scripts for running pipeline stages
├── src/
│   ├── retrieval/            # BM25, dense, and hybrid retriever
│   ├── generation/           # LLM inference + RAG pipeline
│   ├── evaluation/           # Accuracy, robustness metrics
│   └── utils/                # Data loading, logging, helpers
└── tests/                    # Unit tests
```

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/legal-qa-rag.git
cd legal-qa-rag
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python scripts/prepare_data.py \
  --input data/raw/obliqa.json \
  --output data/processed/ \
  --mcq_sizes 4 20
```

### 3. Run Retrieval

```bash
python scripts/run_retrieval.py \
  --config configs/retrieval_hybrid.yaml \
  --split test
```

### 4. Run RAG Inference

```bash
python scripts/run_rag.py \
  --config configs/rag_pipeline.yaml \
  --model phi4-mini \
  --mcq_size 20
```

### 5. Evaluate

```bash
python scripts/evaluate.py \
  --predictions results/predictions.json \
  --split test
```

---

## 🔬 Research Pipeline

```
Raw Legal Docs
      │
      ▼
┌─────────────────┐
│  Hybrid Retriever│  ← BM25 + BGE-M3
└────────┬────────┘
         │ Top-K Passages
         ▼
┌─────────────────┐
│   RAG Pipeline  │  ← Context + MCQ Prompt
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Inference │  ← Phi-4-mini / Gemma-3 / Nemotron
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Fine-Tuning   │  ← LoRA/QLoRA on reasoning traces
│  (LoRA/QLoRA)   │
└────────┬────────┘
         │
         ▼
   Evaluation
  (MCQ-4 / MCQ-20)
```

---

## 📊 Baseline Results

| Model        | MCQ-4 Acc. | MCQ-20 Acc. | Notes                    |
|-------------|------------|-------------|--------------------------|
| Phi-4-mini  | ~90%       | ~60%        | Sensitive to prompts     |
| Gemma-3 4B  | ~88%       | ~55%        | Less robust to variation |
| Nemotron-mini| ~87%      | ~58%        | Stable across prompts    |

> MCQ-20 results show significant performance drop — motivation for hybrid retrieval + fine-tuning.

---

## 🧪 Prompt Strategies

Three prompt variants are supported:

- **Baseline**: Standard instruction + options
- **Adversarial**: Distractor-heavy framing
- **Auditor-style**: Formal regulatory tone

See `configs/prompts.yaml` for templates.

---

## ⚙️ Configuration

All experiment parameters are controlled via YAML configs:

```yaml
# configs/rag_pipeline.yaml
model:
  name: "phi4-mini"
  max_tokens: 512

retrieval:
  type: "hybrid"         # bm25 | dense | hybrid
  top_k: 5
  bm25_weight: 0.4
  dense_weight: 0.6
  dense_model: "BAAI/bge-m3"

prompt:
  style: "baseline"      # baseline | adversarial | auditor

mcq:
  size: 20               # 4 or 20
```

---

## 🔧 Fine-Tuning (LoRA/QLoRA)

```bash
python scripts/finetune.py \
  --config configs/finetune_lora.yaml \
  --base_model deepseek-r1 \
  --data data/processed/train_with_reasoning.json
```

Training uses:
- **LoRA rank**: 16
- **QLoRA**: 4-bit quantization
- **Data**: ~22K legal MCQs with reasoning traces

---

## 📁 Dataset

This project uses **ObliQA** (Obligation-based QA), a legal regulatory QA dataset.

- MCQ-4: Standard 4-choice questions
- MCQ-20: Extended 20-choice (harder, more realistic)

Place your dataset in `data/raw/` before running scripts.

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@misc{yourname2025legalqa,
  title     = {Legal QA with Hybrid RAG and Fine-tuned LLMs},
  author    = {Your Name},
  year      = {2025},
  note      = {PhD Research, University Name}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

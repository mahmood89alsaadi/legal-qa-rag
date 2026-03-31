# ⚖️ Legal QA with RAG + LLMs

> An open-source pipeline for legal regulatory question answering using Retrieval-Augmented Generation and fine-tuned LLMs. Tested on 5 models across 22,295 real legal questions. Runs fully locally — no API key, no cloud costs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![GitHub Stars](https://img.shields.io/github/stars/mahmood89alsaadi/legal-qa-rag)](https://github.com/mahmood89alsaadi/legal-qa-rag)

---

## 🎯 What this does

Standard LLMs hallucinate on legal questions. This project fixes that by:

1. **Retrieving** relevant legal passages using BM25 + BGE-M3 hybrid search
2. **Injecting** them into the prompt using RAG
3. **Evaluating** across 5 open-source LLMs on real regulatory MCQs
4. **Fine-tuning** with LoRA/QLoRA on 22K legal questions with reasoning traces

---

## 📊 Real Results (ObliQA dataset, n=50)

| Model         | MCQ-4  | MCQ-20 | Drop  |
|---------------|--------|--------|-------|
| phi4-mini     | 92.0%  | 96.0%  | -4.0% |
| gemma3:4b     | 92.0%  | 84.0%  | +8.0% |
| nemotron-mini | 72.0%  | 78.0%  | -6.0% |
| mistral       | 92.0%  | 88.0%  | +4.0% |
| llama3.1:8b   | 90.0%  | 92.0%  | -2.0% |

> MCQ-20 is significantly harder — 20 answer options instead of 4. Reducing this performance drop is the core research contribution.

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/mahmood89alsaadi/legal-qa-rag.git
cd legal-qa-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama and pull a model

Download Ollama from https://ollama.com then run:
```bash
ollama pull phi4-mini
```

### 4. Run the demo (no dataset needed)
```bash
python demo_runner.py
```

---

## 🔬 Research Pipeline
```
Legal Documents
      ↓
Hybrid Retrieval (BM25 + BGE-M3)
      ↓
RAG Pipeline (context + question + options)
      ↓
LLM Inference (phi4-mini / gemma3 / llama3.1)
      ↓
Evaluation (MCQ-4 vs MCQ-20 accuracy drop)
```

---

## 🧩 Three prompt styles

| Style       | Description                        |
|-------------|-----------------------------------|
| Baseline    | Standard legal expert prompt      |
| Adversarial | Highlights misleading distractors |
| Auditor     | Formal regulatory compliance tone |

---

## 📁 Project Structure
```
legal-qa-rag/
├── configs/                  # YAML configs for models and retrieval
├── scripts/
│   ├── prepare_data.py       # MCQ-4 to MCQ-20 expansion
│   ├── run_rag.py            # Run RAG pipeline
│   └── finetune.py           # LoRA/QLoRA fine-tuning
├── src/
│   ├── retrieval/            # BM25 + BGE-M3 hybrid retriever
│   ├── generation/           # RAG pipeline
│   ├── evaluation/           # Accuracy and robustness metrics
│   └── utils/                # Prompt builder (3 styles)
├── tests/                    # Unit tests
└── demo_runner.py            # Run everything with no setup
```

---

## 🖥️ Requirements

- Python 3.10+
- Ollama (for local LLM inference)
- 8GB+ RAM (16GB recommended)
- NVIDIA GPU optional but recommended

---

## 👥 Who is this for?

- **Researchers** — reproduce baselines, compare models, extend to new domains
- **Legal tech developers** — plug in your own corpus and deploy a compliance QA system
- **PhD students** — fork and adapt to Arabic, French, medical or financial regulations
- **Law students** — practice regulatory MCQs with AI-powered explanations
- **Anyone** — runs fully locally, no API key, no cloud costs

---

## 🤝 Contributing

Pull requests welcome! Areas that need help:

- Adding new legal datasets
- Supporting more languages (Arabic, French, German)
- Improving retrieval with better embeddings
- Building a simple web interface

---

## 📜 Citation

If you use this in your research please cite:
```bibtex
@misc{alsaadi2025legalqa,
  title  = {Legal QA with Hybrid RAG and Fine-tuned LLMs},
  author = {Alsaadi, Mahmood},
  year   = {2025},
  url    = {https://github.com/mahmood89alsaadi/legal-qa-rag},
  note   = {PhD Research}
}
```

---

## 📄 License

MIT — free to use for research and commercial projects.

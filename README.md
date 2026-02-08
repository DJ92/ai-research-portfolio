# AI Research Portfolio

A collection of projects demonstrating practical AI research skills, with focus on LLM capabilities, evaluation, and production-ready implementations.

## 🎯 Purpose

This portfolio showcases:
- **Rigorous Evaluation**: Systematic measurement of AI system performance
- **Tool Use & Agents**: Function calling, planning, and autonomous behavior
- **Prompt Engineering**: Techniques for reliable LLM outputs
- **Production Systems**: RAG, guardrails, and monitoring
- **Research Mindset**: Reproducible experiments, failure analysis, baselines

## 📂 Projects

### 1. LLM Evaluation Framework
**Status**: ✅ Complete

A comprehensive framework for evaluating LLM outputs across multiple dimensions:
- Automated metrics (BLEU, ROUGE, semantic similarity)
- LLM-as-judge with multi-criteria evaluation
- Unified API client with cost tracking
- Production-ready evaluation patterns

**Tech**: Python, OpenAI/Anthropic APIs, sentence-transformers, pytest

**Highlights**:
- 87% test coverage
- Judge-human correlation: 0.82 (ρ)
- Cost optimization strategies included

[→ View Project](./01-llm-evaluation/) | [📝 Blog Post](https://dj92.github.io/interview-notes/notes/llm-evaluation/)

---

### 2. Tool Use & Function Calling
**Status**: ✅ Complete

Reliable function calling framework with agent patterns:
- Tool registry with schema validation for Anthropic/OpenAI
- ReAct agent implementation (Reason + Act pattern)
- Error handling and parameter validation
- Multi-step reasoning capabilities

**Tech**: Python, Claude API, pydantic, pytest

**Highlights**:
- 94% tool selection accuracy (ReAct + Sonnet)
- Comprehensive test suite
- Production-ready error recovery

[→ View Project](./02-tool-use/)

---

### 3. Production RAG System
**Status**: ✅ Complete

End-to-end RAG implementation with comprehensive evaluation:
- Multiple chunking strategies (fixed-size, sentence, semantic)
- Unified embedding interface (Sentence-Transformers, OpenAI)
- Retrieval quality metrics (MRR, NDCG, Recall@K)
- Advanced techniques (hybrid search, reranking, compression)

**Tech**: Python, ChromaDB, sentence-transformers, OpenAI

**Highlights**:
- Semantic chunking: 85% MRR@10
- Chunking strategy comparison framework
- Embedding model benchmarks with cost analysis
- Production patterns for latency optimization

[→ View Project](./03-rag-system/)

---

### 4. Prompt Engineering Lab
**Status**: ✅ Complete

Systematic evaluation of prompting techniques with empirical findings:
- Core techniques (zero-shot, few-shot, CoT, self-consistency)
- Quantitative comparison across task types
- Automated prompt optimization
- Best practices from systematic experiments

**Tech**: Python, Anthropic Claude, empirical evaluation

**Highlights**:
- Few-shot optimal: 3-5 examples (91% accuracy)
- CoT for math: +36% improvement (78% accuracy)
- Temperature guidelines: T=0 deterministic, T=0.7+ creative
- Self-consistency: +6% for 5x cost

[→ View Project](./04-prompt-engineering/)

---

### 5. Agent Safety & Guardrails
**Status**: ✅ Complete

Production-ready safety mechanisms for LLM agents:
- Prompt injection detection (94% precision, 88% recall)
- Multi-layer defense strategy (reduces attacks by 73%)
- Behavioral constraints (rate limiting, approval workflows)
- Red team testing methodology

**Tech**: Python, regex, spaCy, production monitoring

**Highlights**:
- Defense-in-depth: 4.6% attack success (down from 67%)
- Injection detection: Pattern + LLM ensemble
- Safety overhead: +47% latency for 95%+ protection
- Red team benchmarks across 6 attack types

[→ View Project](./05-agent-safety/)

---

## 🔄 Iterative Enhancements

**Recent Improvements**:

- **Data Pipeline** (01-llm-evaluation): Parallel batch processing with checkpointing, 10x faster evaluation
- **CUDA Optimization** (03-rag-system): GPU-accelerated embeddings, 10-15x speedup with FP16

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-black?logo=anthropic&logoColor=white)

## 📊 Evaluation Philosophy

All projects follow these principles:
1. **Quantitative Metrics**: Clear, measurable success criteria
2. **Baselines**: Compare against simple baselines and SOTA
3. **Failure Analysis**: Document and analyze failure modes
4. **Reproducibility**: All experiments have fixed seeds and configs
5. **Cost Tracking**: Monitor API costs and latency

## 🚀 Getting Started

Each project has its own README with:
- Problem statement and motivation
- Architecture overview
- Setup instructions
- Usage examples
- Evaluation results

## 📝 Blog

I write about these projects and AI research topics on my [technical blog](https://dj92.github.io/interview-notes).

## 📫 Contact

- Email: joshidheeraj1992@gmail.com
- GitHub: [@DJ92](https://github.com/DJ92)
- Blog: [dj92.github.io/interview-notes](https://dj92.github.io/interview-notes)

---

*Built with curiosity about AI capabilities and limitations. All projects prioritize measurement, reliability, and understanding over chasing benchmarks.*

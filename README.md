# AI Research Portfolio

A collection of 12 projects demonstrating **both applied AI research skills and deep foundational knowledge**, spanning from production LLM systems to transformer implementations from scratch.

## 🎯 Purpose

This portfolio showcases:
- **Foundational Knowledge**: Transformers, pre-training, post-training (RLHF/DPO), parameter-efficient tuning
- **Rigorous Evaluation**: Systematic measurement of AI system performance
- **Tool Use & Agents**: Function calling, planning, and autonomous behavior
- **Alignment Research**: Constitutional AI, preference learning, safety mechanisms
- **Production Systems**: RAG, guardrails, and monitoring
- **Research Mindset**: Reproducible experiments, failure analysis, baselines

**Portfolio Structure**: Projects 1-8 focus on applied LLM research (evaluation, agents, safety), while projects 9-12 demonstrate foundational transformer knowledge (architecture, pre-training, alignment techniques, efficiency)

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

### 6. Constitutional AI & Preference Learning
**Status**: ✅ Complete

Implementation of Anthropic's Constitutional AI with critique-revision loop:
- Constitutional principles (harmlessness, helpfulness, honesty, respect)
- Critique-revision loop with iterative refinement
- Preference learning from AI feedback (RLHF simulation)
- Comparison of helpful-only vs helpful+harmless models

**Tech**: Python, Anthropic API, preference modeling

**Highlights**:
- 92% test coverage
- 112% improvement in harmlessness scores
- 82% preference model accuracy
- 73% RLHF improvement rate

[→ View Project](./06-constitutional-ai/)

---

### 7. Chain-of-Thought Faithfulness Analysis
**Status**: ✅ Complete

Analyzes whether CoT reasoning actually drives answers or is post-hoc rationalization:
- CoT parser extracting structured reasoning steps
- Faithfulness analysis with multi-signal detection
- Counterfactual interventions testing reasoning necessity
- Failure mode classification (6 types)

**Tech**: Python, counterfactual analysis, pattern detection

**Highlights**:
- 91% test coverage
- 94% precision in detecting unfaithful reasoning
- 67% of CoT responses show faithful reasoning
- 84% post-hoc rationalization detection accuracy

[→ View Project](./07-cot-faithfulness/)

---

### 8. Model Interpretability Toolkit
**Status**: ✅ Complete

Tools for understanding LLM internals through multiple analysis methods:
- Attention pattern analysis (token-to-token focus, head specialization)
- Logit attribution (input influence on output)
- Perplexity analysis (uncertainty, difficulty proxy)
- Uncertainty quantification (entropy, calibration)
- Semantic clustering (embeddings, concept analysis)

**Tech**: Python, NumPy, visualization tools

**Highlights**:
- 88% test coverage
- PPL-accuracy correlation: r = -0.78
- 82% attribution precision
- Calibration error: 0.15 ECE
- Attention ≠ causation insights

[→ View Project](./08-interpretability/)

---

## 🧠 Foundational Knowledge Projects

The following projects demonstrate deep understanding of transformer architectures, training dynamics, and modern LLM techniques:

---

### 9. Transformer Architecture from Scratch
**Status**: ✅ Complete

Complete GPT-style decoder-only transformer implementation with architectural variants:
- Self-attention mechanism (scaled dot-product, multi-head)
- Positional encodings (sinusoidal, learned, RoPE)
- Transformer blocks (encoder, decoder, encoder-decoder)
- Full GPT model with autoregressive generation
- Attention visualization and analysis tools

**Tech**: PyTorch, attention mechanisms, positional encodings

**Highlights**:
- 38 tests passing, 74% coverage
- Three positional encoding variants compared
- Generation with temperature, top-k, top-p sampling
- Attention as soft dictionary lookup explanation
- O(n²) complexity analysis with solutions

[→ View Project](./09-transformer-architecture/)

---

### 10. Pre-training Techniques (CLM & MLM)
**Status**: ✅ Complete

Implementation and comparison of core pre-training objectives:
- Causal Language Modeling (GPT-style next-token prediction)
- Masked Language Modeling (BERT-style with 80/10/10 strategy)
- Training utilities (warmup, cosine decay, gradient accumulation)
- Perplexity tracking and learning curves
- Data pipeline for pre-training

**Tech**: PyTorch, training loops, optimization

**Highlights**:
- 11 tests passing, 87% coverage
- CLM vs MLM objective comparison
- Warmup + cosine decay scheduling
- 80/10/10 masking strategy (80% [MASK], 10% random, 10% unchanged)
- Perplexity and accuracy metrics

[→ View Project](./10-llm-pre-training/)

---

### 11. Post-Training Methods (SFT, RLHF, DPO)
**Status**: ✅ Complete

Comprehensive implementation of alignment and post-training techniques:
- Supervised Fine-Tuning for instruction following
- Reward Model with Bradley-Terry preference learning
- RLHF framework (reward modeling + PPO)
- Direct Preference Optimization (simpler alternative to RLHF)
- KL divergence penalties to prevent drift

**Tech**: PyTorch, preference modeling, reinforcement learning

**Highlights**:
- 12 tests passing, 86% coverage
- Bradley-Terry model for preferences
- DPO: implicit rewards without separate reward model
- KL penalty to stay close to reference policy
- Comparison of SFT vs RLHF vs DPO tradeoffs

[→ View Project](./11-post-training-methods/)

---

### 12. Parameter-Efficient Fine-Tuning (LoRA & Quantization)
**Status**: ✅ Complete

Efficient fine-tuning techniques for reducing memory and compute:
- LoRA (Low-Rank Adaptation) with configurable rank
- 4-bit and 8-bit quantization
- Weight merging for inference optimization
- Memory footprint analysis
- Parameter efficiency comparisons

**Tech**: PyTorch, low-rank adaptation, quantization

**Highlights**:
- 8 tests passing, 81% coverage
- LoRA: ~99% parameter reduction (h = W₀x + α/r·BAx)
- 8-bit quantization: 4× compression, 4-bit: 8× compression
- Weight merging for zero-overhead inference
- Only 0.1% parameters trainable with LoRA

[→ View Project](./12-parameter-efficient-tuning/)

---

## 🔄 Iterative Enhancements

**Recent Improvements**:

- **Data Pipeline** (01-llm-evaluation): Parallel batch processing with checkpointing, 10x faster evaluation
- **CUDA Optimization** (03-rag-system): GPU-accelerated embeddings, 10-15x speedup with FP16
- **Foundational Projects** (09-12): Added transformer architecture, pre-training, post-training, and PEFT implementations to demonstrate deep ML knowledge

---

## 📈 Portfolio Statistics

- **12 Total Projects**: 8 applied research + 4 foundational implementations
- **Average Test Coverage**: 85%+ across all projects
- **Lines of Code**: ~5000+ lines of research-grade PyTorch/Python
- **Comprehensive Documentation**: Each project has detailed README (150-370 lines)
- **Research Depth**: From production systems → transformer internals → training dynamics → alignment techniques

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-black?logo=anthropic&logoColor=white)

**Core Technologies**:
- **ML Frameworks**: PyTorch, Transformers, sentence-transformers
- **LLM APIs**: OpenAI GPT-4, Anthropic Claude 3.5
- **Vector DBs**: ChromaDB, FAISS
- **Testing**: pytest (85%+ coverage standard)
- **Analysis**: NumPy, attention visualization, perplexity tracking

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

## 🎓 Research Progression

This portfolio demonstrates understanding across the full AI research stack:

**Production & Applied Research** (Projects 1-8):
- Evaluation methodologies → Tool use → RAG systems → Prompt engineering
- Safety & guardrails → Constitutional AI → CoT faithfulness → Interpretability

**Foundational Knowledge** (Projects 9-12):
- Transformer architecture → Pre-training objectives → Post-training alignment → Parameter efficiency

**Key Insight**: Understanding both "how to use LLMs effectively" AND "how LLMs work internally" - from production systems to training dynamics to alignment techniques.

---

*Built with curiosity about AI capabilities and limitations. All projects prioritize measurement, reproducibility, and deep understanding over chasing benchmarks.*

# Post-Training Methods: SFT, RLHF, and DPO

Implementing alignment techniques that transform pre-trained language models into helpful, harmless, and honest assistants.

## 📖 Overview

This project implements **post-training methods** that align language models with human preferences. These techniques bridge the gap between raw pre-trained models and production assistants.

### Research Questions

1. **Why does RLHF work?** How reward signals improve model behavior beyond supervised learning
2. **What are the tradeoffs?** SFT vs RLHF vs DPO in terms of complexity, stability, and quality
3. **How to prevent reward hacking?** KL penalties and their impact on alignment
4. **Can we skip the reward model?** DPO as a simpler alternative to RLHF

## 🎯 Post-Training Pipeline

### Traditional Path (RLHF)
```
Pre-trained Model
    ↓
Supervised Fine-Tuning (SFT)
    ↓
Reward Model Training
    ↓
PPO Optimization (RLHF)
    ↓
Aligned Model
```

### Modern Alternative (DPO)
```
Pre-trained Model
    ↓
Supervised Fine-Tuning (SFT)
    ↓
Direct Preference Optimization (DPO)
    ↓
Aligned Model
```

## 🔧 Method 1: Supervised Fine-Tuning (SFT)

**Goal:** Teach model to follow instructions and respond in desired format

**Data Format:**
```json
{
  "instruction": "Write a poem about machine learning",
  "response": "In circuits deep where patterns hide,\n           Neural networks learn with pride..."
}
```

**Training Objective:** Standard cross-entropy on high-quality responses

**Why it works:**
- Conditions model on instruction-following examples
- Shifts distribution toward helpful responses
- Provides basic alignment and format

**Limitations:**
- No notion of "better" vs "worse" responses
- Can't optimize for preferences beyond training data
- May produce plausible but incorrect answers

**Implementation:**
```python
from sft import SupervisedFineTuner

# Prepare instruction-response pairs
sft_trainer = SupervisedFineTuner(
    model=pretrained_model,
    dataset=instruction_dataset,
    learning_rate=1e-5,
    batch_size=16
)

sft_model = sft_trainer.train(epochs=3)
```

## 🎯 Method 2: RLHF (Reinforcement Learning from Human Feedback)

**Three-stage process:**

### Stage 1: Train Reward Model

**Goal:** Learn to score model outputs based on human preferences

**Data Format:**
```json
{
  "prompt": "Explain quantum computing",
  "response_chosen": "Quantum computing uses quantum bits...",
  "response_rejected": "Quantum computing is magic..."
}
```

**Training Objective:** Bradley-Terry model
```
Loss = -log(σ(r_chosen - r_rejected))

Where σ = sigmoid function
```

**Why this works:**
- Reward model learns to predict human preferences
- Can generalize beyond training preferences
- Provides dense reward signal for RL

### Stage 2: PPO Optimization

**Goal:** Maximize reward while staying close to SFT model

**Objective:**
```
maximize: E[reward(x, y)] - β * KL(π_θ || π_ref)

Where:
- π_θ = policy being trained
- π_ref = reference model (SFT)
- β = KL penalty coefficient
```

**Why KL penalty matters:**
- Prevents reward hacking (exploiting reward model)
- Maintains coherent language generation
- Balances optimization vs safety

**PPO Algorithm:**
1. Sample prompts from dataset
2. Generate responses with current policy
3. Score with reward model
4. Compute advantages
5. Update policy with clipped objective
6. Repeat

### Complete RLHF Pipeline

```python
from rlhf import RewardModel, PPOTrainer

# Stage 1: Train reward model
reward_model = RewardModel(base_model, d_model=768)
reward_trainer = RewardTrainer(reward_model, preference_dataset)
reward_model = reward_trainer.train()

# Stage 2: PPO optimization
ppo_trainer = PPOTrainer(
    policy=sft_model,
    reward_model=reward_model,
    ref_model=sft_model.clone(),  # Frozen reference
    kl_coef=0.1,  # KL penalty
    learning_rate=1e-6
)

aligned_model = ppo_trainer.train(num_steps=10000)
```

## 🚀 Method 3: Direct Preference Optimization (DPO)

**Key insight:** Skip the reward model and optimize policy directly from preferences!

**Objective (simplified):**
```
maximize: E[log σ(β * log(π_θ(y_w|x) / π_ref(y_w|x))
                    - β * log(π_θ(y_l|x) / π_ref(y_l|x)))]

Where:
- y_w = preferred response
- y_l = rejected response
- β = temperature parameter
```

**Why this works:**
- DPO implicitly defines a reward function
- Equivalent to RLHF under certain conditions
- Much simpler to implement (no RL, no reward model)
- More stable training

**Advantages over RLHF:**
- ✅ No reward model needed (saves memory/compute)
- ✅ No RL instability (supervised learning)
- ✅ Fewer hyperparameters
- ✅ Often works as well or better

**Implementation:**
```python
from dpo import DirectPreferenceOptimization

dpo_trainer = DirectPreferenceOptimization(
    model=sft_model,
    ref_model=sft_model.clone(),  # Frozen reference
    preference_dataset=preference_dataset,
    beta=0.1,  # Temperature
    learning_rate=5e-7
)

aligned_model = dpo_trainer.train(epochs=1)
```

## 📊 Method Comparison

| Dimension | SFT | RLHF | DPO |
|-----------|-----|------|-----|
| **Complexity** | Low | High | Medium |
| **Training stability** | High | Low | High |
| **Compute cost** | Low | High | Medium |
| **Quality** | Good | Best | Very Good |
| **Use case** | Basic alignment | High-quality assistants | Practical alignment |
| **Requires** | Demonstrations | Preferences + RL | Preferences |

## 🔬 Experiments

### Experiment 1: SFT Baseline

**Setup:** Fine-tune GPT-small on 10K instruction-response pairs

**Metrics:**
- Instruction following accuracy
- Response format compliance
- Perplexity on held-out instructions

**Expected:** Model learns to follow format but may hallucinate

### Experiment 2: RLHF vs DPO

**Setup:** Train both methods on same preference data

**Compare:**
- Final reward model scores
- Win rate vs SFT baseline (human eval)
- Training stability (loss curves)
- Compute time and memory usage

**Expected:** DPO matches RLHF quality with 3x less compute

### Experiment 3: KL Penalty Ablation

**Question:** How does KL coefficient affect alignment?

**Setup:** Train RLHF/DPO with β ∈ {0.01, 0.05, 0.1, 0.5}

**Metrics:**
- Reward model score
- KL divergence from reference
- Response diversity
- Coherence (human eval)

**Expected:**
- Too low β → reward hacking, incoherent text
- Too high β → no improvement over SFT
- Optimal β ≈ 0.1

### Experiment 4: Reward Model Quality

**Question:** How does reward model accuracy affect RLHF?

**Setup:** Train reward models with varying amounts of preference data

**Expected:** Better reward model → better RLHF, but DPO more robust

## 🚀 Usage

### Quick Start: SFT

```python
from sft import SupervisedFineTuner
from data import InstructionDataset

# Load instruction dataset
dataset = InstructionDataset("data/instructions.json")

# Fine-tune
trainer = SupervisedFineTuner(model, dataset)
sft_model = trainer.train(epochs=3)
```

### Complete RLHF Pipeline

```python
# 1. SFT
sft_model = train_sft(model, instruction_dataset)

# 2. Train reward model
reward_model = train_reward_model(preference_dataset)

# 3. PPO
aligned_model = train_ppo(sft_model, reward_model, prompts)
```

### DPO (Simpler Alternative)

```python
# 1. SFT
sft_model = train_sft(model, instruction_dataset)

# 2. DPO (no reward model needed!)
aligned_model = train_dpo(sft_model, preference_dataset)
```

## 🧪 Testing

Run comprehensive test suite:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Test Coverage:**
- ✅ SFT loss computation
- ✅ Reward model pairwise ranking
- ✅ PPO advantage calculation
- ✅ DPO loss computation
- ✅ KL divergence computation
- ✅ Gradient flow through all methods

Expected coverage: **90%+**

## 📚 Key Insights

### 1. Why RLHF Works

**The optimization problem:**
- Pre-training: Predict next token (unsupervised)
- SFT: Imitate demonstrations (supervised)
- RLHF: Maximize reward (reinforcement learning)

**What RLHF adds:**
- Optimization for preferences (not just imitation)
- Ability to discover better responses than training data
- Explicit reward signal guides improvement

### 2. The Reward Hacking Problem

**What can go wrong:**
```
High reward but nonsensical:
"Sure! Here's a poem: AAAAA... [repeated 1000 times]"

Why? Reward model was fooled by length + enthusiasm
```

**Solution:** KL penalty keeps policy close to reference

```
Objective = Reward - β * KL(policy || reference)
             ↑            ↑
         optimize      stay reasonable
```

### 3. DPO: Eliminating the Reward Model

**Key insight from DPO paper:**
- RLHF optimal policy has closed form: `π*(y|x) ∝ π_ref(y|x) * exp(r(x,y)/β)`
- Can rearrange to get reward: `r(x,y) = β * log(π*/π_ref) + const`
- Substitute into preference loss → direct policy optimization!

**Result:** Same objective as RLHF but no reward model needed

### 4. Constitutional AI Connection

This project connects to **Project 06: Constitutional AI**:
- SFT/RLHF/DPO are the *mechanisms*
- Constitutional principles are the *values*
- Together: scalable oversight for alignment

## 🔮 Future Work

- [ ] Implement reward model ensembles (reduce reward hacking)
- [ ] Add RLAIF (AI feedback instead of human)
- [ ] Implement rejection sampling (simpler than PPO)
- [ ] Add KTO (Kahneman-Tversky Optimization)
- [ ] Multi-objective RLHF (helpful + harmless + honest)
- [ ] Online vs offline RL comparison

## 📖 References

- **InstructGPT** (Ouyang et al., 2022) - RLHF for ChatGPT
- **DPO** (Rafailov et al., 2023) - Direct Preference Optimization
- **Constitutional AI** (Bai et al., 2022) - Principles-based alignment
- **PPO** (Schulman et al., 2017) - Proximal Policy Optimization

---

*This project demonstrates deep understanding of alignment techniques and their tradeoffs for building safe, helpful AI systems.*

# Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/ReasoningTransferability)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/ReasoningTransfer/Transferability-of-LLM-Reasoning)

**TL;DR**: We find that while supervised fine-tuning (SFT) on math data improves math reasoning but hurts general capabilities, reinforcement learning (RL) achieves strong math performance while preserving—and even improving—broader domain performance.

**We will release our code very soon! Stay tuned!**

## 🔍 Overview

As large language models (LLMs) rapidly advance on mathematical reasoning benchmarks like MATH and AIME, a critical question emerges: **Do these gains reflect broader problem-solving ability or just narrow overfitting?**

This repository contains the code, data, and evaluation framework for our comprehensive study evaluating over 20 open-weight reasoning-tuned models across math, scientific QA, agent planning, coding, and standard instruction-following tasks.

### 🎯 Main Research Question
**Does improved mathematical reasoning transfer to general LLM capabilities?**

## 🔑 Key Findings

### 1. Training Method Matters More Than Model Size
- **RL-tuned models** generalize well across domains
- **SFT-tuned models** suffer catastrophic forgetting on non-reasoning tasks
- This pattern holds consistently across model families (1.5B to 32B parameters)

### 2. Transferability Index Results
![Transferability Results](assets/transferability_comparison.png)

| Method | Math Reasoning | Other Reasoning | Non-Reasoning |
|--------|---------------|----------------|---------------|
| **RL** | ✅ Strong gains | ✅ Positive transfer | ✅ Preserved/improved |
| **SFT** | ✅ Strong gains | ⚠️ Limited transfer | ❌ Performance degradation |

### 3. Internal Analysis Reveals Why
- **PCA Analysis**: RL preserves general-domain representation structure
- **Token Distribution**: RL selectively shifts task-relevant tokens, SFT perturbs many irrelevant ones
- **KL Divergence**: RL shows minimal distribution drift from base models

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/ReasoningTransfer/Transferability-of-LLM-Reasoning.git
cd Transferability-of-LLM-Reasoning

# Create conda environment
conda create -n reasoning-transfer python=3.10
conda activate reasoning-transfer

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## ⚡ Quick Start


## 📊 Evaluation

### Benchmark Categories

| Category | Benchmarks | Description |
|----------|------------|-------------|
| **Math Reasoning** | MATH-500, AIME24/25, OlympiadBench | Pure mathematical problem solving |
| **Other Reasoning** | GPQA-Diamond, LiveCodeBench, ACPBench, HeadQA | Scientific QA, coding, planning |
| **Non-Reasoning** | CoQA, IFEval, HaluEval, MC-TACO | Conversational QA, instruction following |

### Run Full Evaluation Suite

```bash
# Evaluate a single model



## Analysis Tools

Reproduce our internal analyses:

```bash
# PCA shift analysis
python analysis/analyse_PCA_Shift.py \
    --base_model Qwen3-14B-Base \
    --fine_tuned_models models/unireason_*/ \
    --task_type math,other_reasoning,non_reasoning \
    --k 100(query numbers)

# Token distribution analysis  
python analysis/token_analysis.py \
    --models models/unireason_*/ \
    --base_model Qwen3-14B-Base \
    --generate_wordclouds
```

## 📊 Pre-trained Models

Access our trained models via Hugging Face:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# UniReason SFT model
model = AutoModelForCausalLM.from_pretrained("ReasoningTransferability/UniReason-Qwen3-14B-SFT")

# UniReason RL model  
model = AutoModelForCausalLM.from_pretrained("ReasoningTransferability/UniReason-Qwen3-14B-RL")

tokenizer = AutoTokenizer.from_pretrained("ReasoningTransferability/UniReason-Qwen3-14B-RL")
```


## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{huan2025math,
  title={Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning},
  author={Huan, Maggie and Li, Yuetai and Zheng, Tuney and Xu, Xiaoyu and Kim, Seungone and Du, Minxin and Poovendran, Radha and Neubig, Graham and Yue, Xiang},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

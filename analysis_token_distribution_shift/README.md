# Token Distribution Shift Analysis

This folder contains tools for analyzing token-level distribution shifts between different LLMs. The analysis focuses on calculating token-level logits, ranks, and KL divergence to understand how the token-level distribution shift after RL/SFT.

## Overview

1. **Token-Level Logits and Ranks Calculator** (`calculate_logits_and_ranks_token_level.py`)
2. **KL Divergence Calculator** (`calculate_KL.py`)
3. **Automated Pipeline** (`token_level_logits_and_ranks.sh`)





## Usage

### 1. Token Loss and Rank Calculate

Calculate token-level logits and ranks for specific models:

```bash
python calculate_logits_and_ranks_token_level.py \
    --dataset finetuned_model_response/UniReason-SFT_ifeval.jsonl \
    --models "ReasoningTransferability/UniReason-Qwen3-14B-think-SFT,Qwen/Qwen3-14B-base" \
    --batch_size 1 \
    --output_dir ppl_data/UniReason-SFT \
```

**Parameters:**
- `--dataset`: Path to the JSONL dataset file
- `--models`: Comma-separated list of model names
- `--batch_size`: Number of samples per batch (default: 1)
- `--output_dir`: Output directory for results

### 2. KL Divergence Calculation

Calculate KL divergence between models:

```bash
python calculate_KL.py \
    --data_dir ppl_data/UniReason-SFT \
    --target_gen_model UniReason-SFT \
    --output_file kl_divergence_results.json
```

**Parameters:**
- `--data_dir`: Directory containing token loss data
- `--target_gen_model`: Target generation model name
- `--output_file`: Output file path (optional)



Or you can directly run the following automated pipeline
### 3. Automated Pipeline

Run the complete analysis pipeline:

```bash
bash token_level_logits_and_ranks.sh
```

This script will:
1. Process all configured experiments and tasks
2. Calculate token-level statistics for each model
3. Compute KL divergence between models
4. Save results to organized output directories


## Configuration

### Experiment Configuration

Modify the `EXPERIMENTS` dictionary in `token_level_logits_and_ranks.sh`:

```bash
EXPERIMENTS["experiment_name"]="fine-tuned-model base-model"
```



## License

This project is licensed under the MIT License.
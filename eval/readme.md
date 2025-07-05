# Evaluation

We use [EvalChemy](https://github.com/mlfoundations/evalchemy) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate the performance of the fine-tuned models.

## Build Environment

**For EvalChemy:**
```bash
cd eval/evalchemy
conda create -n evalchemy
conda activate evalchemy python=3.10
pip install -e .
pip install vllm==0.8.5
```

**For lm-evaluation-harness:**
```bash
cd eval/lm-evaluation-harness
conda create -n lm_eval
conda activate lm_eval
pip install -e .
pip install vllm==0.8.5
```

## Available Tasks

**For EvalChemy:**
- AIME24  
- AIME25  
- MATH500
- GPQADiamond
- LiveCodeBench
- IFEval

**For lm-evaluation-harness:**
- HeadQA
- MC_TACO
- IFEVAL
- CoQA
- HaluEval
- Olympiad  

## Usage

### Configuration

1. **Set up your models**: Edit the `MODELS` array in the script to include your model paths
2. **Set unique names**: Edit the `UNIQUE_NAMES` array to provide nicknames for your models
3. **Configure GPU settings**: Adjust `tensor_parallel_size` and `data_parallel_size` to fit your GPU configuration
4. **Set HuggingFace cache**: Configure `HF_HOME` and `HF_HUB_CACHE` environment variables

### Model Path Format

The `model_path` can be either:
- A local model path: `/path/to/your/local/model`
- A HuggingFace repository name: `"ReasoningTransferability/UniReason-Qwen3-14B-RL"`

### Running Evaluations

The script automatically handles different model configurations:

- **Standard Qwen3-14B models**: Use `max_output_length=32768`
- **Qwen2.5-7B models** (7B variants): Use `max_output_length=4096` for:
  - Qwen2.5-7B-Instruct
  - Qwen2.5-Math-7B-Instruct  
  - Qwen-2.5-Math-7B-SimpleRL-Zoo

Execute the evaluation script:
```bash
bash run_your_model.sh
```

### Script Features

- **Automatic logging**: Creates timestamped log files in the `logs/` directory
- **Model name generation**: Automatically generates unique model names based on path or uses provided nicknames
- **GPU configuration**: Supports multi-GPU setups with configurable parallelization
- **Batch processing**: Evaluates multiple models sequentially
- **Summary generation**: Creates evaluation summaries for analysis

### GPU Settings

Adjust the following parameters in the script based on your hardware:
- `CUDA_VISIBLE_DEVICES`: Specify which GPUs to use
- `gpu_num`: Number of GPUs available
- `tensor_parallel_size`: For model parallelization across GPUs
- `data_parallel_size`: For data parallelization
- `gpu_memory_utilization`: Memory usage ratio (default: 0.85)

### Output

The script generates:
- Individual log files for each model evaluation
- Overall evaluation log with timestamp
- Summary file with consolidated results
- JSON output files in the `logs/` directory

### Example Configuration

```bash
MODELS=(
  "/path/to/your/local/model"
  "huggingface/model-repo-name"
)

UNIQUE_NAMES=(
  "MyModel_v1"
  "BaselineModel"
)
```

## Notes

- For each evaluation, example bash scripts are available in `eval/lm-evaluation-harness/run_your_model.sh` and `eval/evalchemy/run_your_model.sh`
- Change your model and path in the script to run evaluations
- The script uses vLLM for efficient inference with automatic batch sizing
- All outputs are saved with timestamps to prevent conflicts
#!/bin/bash

mkdir -p logs

export HF_HOME=""
export HF_HUB_CACHE=""

MODELS=(
  "your_model_1_here"
  "your_model_2_here"
)

UNIQUE_NAMES=(
  "your_model_1_'s_nickname_here"
  "your_model_2_'s_nickname_here"
)

EVAL_TASKS=(
  "AIME24"
  "AIME25"
  "MATH500"
  "GPQADiamond"
  "LiveCodeBench"
  "IFEval"
)
TASKS_STR=$(IFS=, ; echo "${EVAL_TASKS[*]}")

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_num=8
default_max_output_length=32768

set -e

overall_log="logs/eval_$(date +"%Y%m%d_%H%M%S").log"

function get_unique_model_name() {
    local model_path=$1
    local index=$2

    if [[ $index -lt ${#UNIQUE_NAMES[@]} ]]; then
        echo "${UNIQUE_NAMES[$index]}"
    else
        local base_name=$(basename "$model_path")
        local parent_dir=$(basename $(dirname "$model_path"))
        
        if [[ "$parent_dir" != "models" && "$parent_dir" != "saves" ]]; then
            echo "${parent_dir}_${base_name}"
        else
            echo "${base_name}_${index}"
        fi
    fi
}

for i in "${!MODELS[@]}"; do
  MODEL_PATH="${MODELS[$i]}"
  MODEL_NAME=$(get_unique_model_name "$MODEL_PATH" "$i")

  if [[ "$MODEL_PATH" == *"Qwen2.5-7B-Instruct"* ]] || \
     [[ "$MODEL_PATH" == *"Qwen2.5-Math-7B-Instruct"* ]] || \
     [[ "$MODEL_PATH" == *"Qwen-2.5-Math-7B-SimpleRL-Zoo"* ]]; then
    max_output_length=4096
    echo "max_output_length=$max_output_length"
  else
    max_output_length=$default_max_output_length
    echo "max_output_length=$max_output_length"
  fi

  current_time=$(date +"%Y%m%d_%H%M%S")
  log_file="logs/${MODEL_NAME}_eval_${current_time}.log"
  batch_size="auto"

  python -m eval.eval \
    --model vllm \
    --tasks "${TASKS_STR}" \
    --model_args "pretrained=${MODEL_PATH},tensor_parallel_size=${gpu_num},gpu_memory_utilization=0.85,dtype=bfloat16,max_model_len=${max_output_length}"  \
    --batch_size "$batch_size" \
    --output_path logs 2>&1 | tee "$log_file"

done

summary_file="logs/eval_summary_$(date +"%Y%m%d_%H%M%S").txt"
echo "============" >> "$summary_file"
echo "" >> "$summary_file"

for i in "${!MODELS[@]}"; do
  MODEL_PATH="${MODELS[$i]}"
  MODEL_NAME=$(get_unique_model_name "$MODEL_PATH" "$i")

  if [[ "$MODEL_PATH" == *"Qwen2.5-7B-Instruct"* ]] || \
     [[ "$MODEL_PATH" == *"Qwen2.5-Math-7B-Instruct"* ]] || \
     [[ "$MODEL_PATH" == *"Qwen-2.5-Math-7B-SimpleRL-Zoo"* ]]; then
    max_output_length=4096
  else
    max_output_length=$default_max_output_length
  fi
  
done
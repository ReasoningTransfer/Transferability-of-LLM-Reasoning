#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status,
# treat unset variables as an error, and pipefail to catch errors in piped commands.
set -euo pipefail

# ===== Fixed resources and common parameters =====
# Restrict to GPU device ID 2
export CUDA_VISIBLE_DEVICES=2
# Number of samples per task
K=100

# ===== Base model checkpoints =====
base_models=(
  "Qwen/Qwen2.5-7B"           
  "Qwen/Qwen2.5-7B-Instruct"  
)
# ===== Fine-tuned model checkpoints =====
fine_tuned_models=(
  "hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo"  
  "simplescaling/s1.1-7B"               
)

# ===== List of evaluation tasks =====
TASKS=(
  MATH500
)

# ===== Analysis scripts to run =====
SCRIPTS=(
  analyse_PCA_Shift.py  # Script for PCA shift analysis
)

# ===== Triple nested loops: model group × script × task =====
for i in "${!base_models[@]}"; do
  BASE_MODEL="${base_models[$i]}"     # Select current base model
  STEP1="${fine_tuned_models[$i]}"    # Corresponding fine-tuned model

  # Display header for clarity
  echo "==========================================="
  echo "🧠 Using MODEL_GROUP $((i+1)) :"
  echo "BASE_MODEL = ${BASE_MODEL}"
  echo "STEP1     = ${STEP1}"
  echo "==========================================="

  # Iterate over each analysis script
  for script in "${SCRIPTS[@]}"; do
    # Iterate over each evaluation task
    for task in "${TASKS[@]}"; do
      echo "▶️  Running ${script} on task_type=${task}"

      # If the script name contains "PCA_", pass the PCA-specific arguments
      if [[ "${script}" == *PCA_* ]]; then
        cmd="python ${script} \
              --base_model \"${BASE_MODEL}\" \
              --fine_tuned_model \"${STEP1}\" \
              --task_type \"${task}\" \
              --k ${K}"
      fi

      # Print and execute the constructed command
      echo "CMD: ${cmd}"
      eval ${cmd}
      echo "✅ Finished ${script} (${task})"
    done
  done
done

echo "🎉 All analyses completed!"  # Final completion message

#!/bin/bash

# Token-Level Logits and Ranks Analysis Pipeline
# This script processes multiple experiments and tasks to calculate token-level statistics
# and KL divergence between different models.

set -e  # Exit on any error

# Configuration
BATCH_SIZE="auto"
TASKS=("ifeval")
# "hendrycks_math_500" "Olympiad" "gpqa_diamond_cot_zeroshot" "coqa"

# Experiment configurations
declare -A EXPERIMENTS
EXPERIMENTS["UniReason-SFT"]="ReasoningTransferability/UniReason-Qwen3-14B-think-SFT Qwen/Qwen3-14B-base"
EXPERIMENTS["UniReason-RL"]="ReasoningTransferability/UniReason-Qwen3-14B-RL Qwen/Qwen3-14B-base"

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if required files exist
check_dependencies() {
    local required_files=("calculate_logits_and_ranks_token_level.py" "calculate_KL.py")
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_message "ERROR: Required file $file not found"
            exit 1
        fi
    done
    
    log_message "All required files found"
}

# Function to create output directories
create_output_dirs() {
    for exp_name in "${!EXPERIMENTS[@]}"; do
        mkdir -p "ppl_data/${exp_name}"
        log_message "Created output directory: ppl_data/${exp_name}"
    done
}

# Main processing function
process_experiments() {
    for exp_name in "${!EXPERIMENTS[@]}"; do
        log_message "Processing experiment: $exp_name"
        
        # Parse models for this experiment
        IFS=' ' read -r -a models <<< "${EXPERIMENTS[$exp_name]}"
        
        # Get the first model as the target generation model
        target_model="${models[0]}"
        log_message "Target model: $target_model"
        
        for task in "${TASKS[@]}"; do
            log_message "Processing task: $task"
            
            # Check if input data file exists
            input_file="finetuned_model_response/${exp_name}_${task}.jsonl"
            if [[ ! -f "$input_file" ]]; then
                log_message "WARNING: Input file $input_file not found, skipping"
                continue
            fi
            
            # Calculate probabilities for all models
            for model in "${models[@]}"; do
                log_message "Calculating token losses for model: $model"
                
                python calculate_logits_and_ranks_token_level.py \
                    --dataset "$input_file" \
                    --models "$model" \
                    --batch_size 1 \
                    --output_dir "ppl_data/${exp_name}" \
                
                if [[ $? -eq 0 ]]; then
                    log_message "Successfully processed $model for task $task"
                else
                    log_message "ERROR: Failed to process $model for task $task"
                fi
            done
        done
        
        # Calculate KL divergence for this experiment
        log_message "Calculating KL divergence for experiment: $exp_name"
        
        python calculate_KL.py \
            --data_dir "ppl_data/${exp_name}" \
            --target_gen_model "$exp_name"
        
        if [[ $? -eq 0 ]]; then
            log_message "Successfully calculated KL divergence for $exp_name"
        else
            log_message "ERROR: Failed to calculate KL divergence for $exp_name"
        fi
        
        log_message "Completed experiment: $exp_name"
        echo "----------------------------------------"
    done
}

# Main execution
main() {
    log_message "Starting token-level analysis pipeline"
    
    # Check dependencies
    check_dependencies
    
    # Create output directories
    create_output_dirs
    
    # Process all experiments
    process_experiments
    
    log_message "Pipeline completed successfully"
}

# Run main function
main "$@"
source your_path/bin/activate
conda activate your_environment

export HF_HOME=""
export HF_ALLOW_CODE_EVAL="1"

cd eval/lm-evaluation-harness/

models=(
    "your_model_1"
    "your_model_2"
)

tasks=(
    "acp_mcq_cot_2shot"
    "headqa_en"
    "halueval"
    "mc_taco"
    "olympiad"
    "hendrycks_math_500"
    "ifeval"

    "openbookqa"
    "race"
    "sciq"
    "logiqa"
    "piqa"
    "AIME"
    "aime25"
    "commonsense_qa"
    "social_iqa"
    "siqa"
)

gpu_num="8"

default_max_model_tokens=32768
default_max_gen_tokens=32768
# default_max_model_tokens=8192
# default_max_gen_tokens=8192

# model arguments
base_model_args="tensor_parallel_size=$gpu_num,data_parallel_size=1,gpu_memory_utilization=0.6,dtype=bfloat16"
batch_size="auto"

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        echo "Running lm_eval with model: $model, task: $task, run: $run"
        
        if [[ "$model" == *"Qwen2.5-7B-Instruct"* ]] || [[ "$model" == *"Qwen2.5-Math-7B-Instruct"* ]] || [[ "$model" == *"Qwen-2.5-Math-7B-SimpleRL-Zoo"* ]]; then
            max_model_tokens=4096
            max_gen_tokens=4096
        else
            max_model_tokens=$default_max_model_tokens
            max_gen_tokens=$default_max_gen_tokens
        fi
        
        model_args="$base_model_args,max_model_len=$max_model_tokens"
        
        run_output_path="${output_path}_run${run}"
        
        lm_eval --model vllm \
            --model_args pretrained=${model},$model_args \
            --gen_kwargs do_sample=False,max_gen_toks=$max_gen_tokens \
            --tasks "$task" \
            --batch_size "$batch_size" \
            --log_samples \
            --trust_remote_code \
            --apply_chat_template \
            --output_path "$run_output_path"
    done
done

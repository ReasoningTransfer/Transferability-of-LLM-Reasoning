# -*- coding: utf-8 -*-
"""
Token-Level Logits and Ranks Calculator

This script calculates the loss for each token in responses from a specified dataset 
using a given model (based on logprob calculations), and records the loss details 
for each token. This version uses the vLLM framework for model inference, directly 
loading all 4 GPUs for parallel processing.

Usage:
    python calculate_logits_and_ranks_token_level.py --dataset <dataset_path> --models <model_names> --output_dir <output_path>
"""

import torch
import json
import math
import argparse
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer
from tqdm import tqdm

# Import vLLM framework
from vllm import LLM, SamplingParams


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_batch_losses_vllm(
    engine: LLM, 
    tokenizer: AutoTokenizer, 
    batch_entries: List[Dict[str, Any]]
) -> List[Optional[Dict[str, Any]]]:
    """
    Use vLLM framework to compute loss for each token in a mini-batch of data.
    
    For each valid sample (sample must contain both "input" and "output" fields),
    concatenate the input and output, calculate the number of tokens in the input part,
    skip the input part tokens when generating results, extract logprob for each token
    in the output part and calculate loss = -logprob, while computing the average loss
    and perplexity (via exp(average loss)) for the output part.
    
    Args:
        engine: vLLM engine instance
        tokenizer: Tokenizer for the model
        batch_entries: List of data entries to process
        
    Returns:
        List of results (None for invalid samples)
    """
    prompts = []
    prompt_lengths = []
    valid_indices = []  # Record the indices of valid samples in the batch

    for idx, entry in enumerate(batch_entries):
        problem = entry.get("input", "")
        solution = entry.get("output", "")
        
        if problem and solution:
            full_text = problem + solution
            prompts.append(full_text)
            
            # Calculate the number of tokens in the input part
            problem_tokens = tokenizer(
                problem, 
                return_tensors="pt", 
                truncation=True, 
                add_special_tokens=False
            )["input_ids"][0]
            prompt_lengths.append(len(problem_tokens))
            valid_indices.append(idx)

    # If no valid samples, return None list directly
    if not prompts:
        return [None] * len(batch_entries)

    final_results = [None] * len(batch_entries)
    
    try:
        # Use vLLM for batch inference
        sampling_params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
        outputs = engine.generate(prompts, sampling_params)

        for batch_idx, output in zip(valid_indices, outputs):
            # Get all prompt_logprobs for this sample
            raw_prompt_logprobs = output.prompt_logprobs
            if raw_prompt_logprobs is None or not isinstance(raw_prompt_logprobs, list):
                logger.warning(f"Sample {batch_idx} prompt_logprobs does not exist or has wrong format")
                final_results[batch_idx] = None
                continue

            # Use the corresponding order of valid_indices to get prompt_length
            prompt_length = prompt_lengths[valid_indices.index(batch_idx)]
            # Answer part logprobs start from prompt_length
            answer_token_entries = raw_prompt_logprobs[prompt_length + 1:]
            
            token_losses = []
            for token_entry in answer_token_entries:
                # Get the candidate logprob with rank==1
                _, logprob_obj = list(token_entry.items())[0]
                chosen_logprob = logprob_obj.logprob
                decoded_token = logprob_obj.decoded_token
                token_rank = logprob_obj.rank

                token_losses.append({
                    "loss": -chosen_logprob, 
                    "token": decoded_token, 
                    "rank": token_rank
                })

            if token_losses:
                token_losses_values = [token_loss["loss"] for token_loss in token_losses]
                avg_loss = sum(token_losses_values) / len(token_losses_values)
                perplexity = math.exp(avg_loss)
                
                result_entry = {
                    "avg_loss": avg_loss,
                    "perplexity": perplexity,
                    "token_losses": token_losses
                }
                final_results[batch_idx] = result_entry
            else:
                final_results[batch_idx] = None
                
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        # Set results to error state for all valid samples in the batch
        for idx in valid_indices:
            final_results[idx] = {
                "avg_loss": 0.0,
                "perplexity": 0.0,
                "token_losses": [],
                "error": str(e)
            }
            logger.warning(f"Sample {idx} processing failed, marked as error and continuing")

    return final_results


def load_dataset(dataset_file: str) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file."""
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
            # Take the first 100 samples for testing
            dataset = dataset[:100]
        logger.info(f"Loaded {len(dataset)} samples from {dataset_file}")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset from {dataset_file}: {str(e)}")
        return []


def process_model(
    model_name: str, 
    dataset: List[Dict[str, Any]], 
    batch_size: int
) -> List[Optional[Dict[str, Any]]]:
    """Process a single model with the given dataset."""
    logger.info(f"Loading model: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        engine = LLM(
            model=model_name, 
            gpu_memory_utilization=0.4, 
            dtype="bfloat16", 
            tensor_parallel_size=4
        )
        logger.info(f"Successfully loaded vLLM model: {model_name}")
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        return [None] * len(dataset)

    results = []
    n_batches = (len(dataset) + batch_size - 1) // batch_size

    with tqdm(total=n_batches, desc=f"{model_name} Progress") as pbar:
        for i in range(0, len(dataset), batch_size):
            try:
                mini_batch = dataset[i:i+batch_size]
                batch_results = compute_batch_losses_vllm(engine, tokenizer, mini_batch)
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}/{n_batches}: {e}")
                # Set results to error state for all samples in this batch
                batch_results = []
                for _ in range(len(mini_batch)):
                    batch_results.append({
                        "avg_loss": 0.0,
                        "perplexity": 0.0,
                        "token_losses": [],
                        "error": str(e)
                    })
                results.extend(batch_results)
                logger.warning(f"Batch {i//batch_size + 1} marked as error and continuing")
            
            pbar.update(1)
            torch.cuda.empty_cache()

    return results


def calculate_and_print_stats(results: List[Optional[Dict[str, Any]]], model_name: str) -> Optional[float]:
    """Calculate and print statistics for the results."""
    valid_perps = [
        res["perplexity"] for res in results 
        if res is not None and res["perplexity"] is not None and not math.isinf(res["perplexity"])
    ]
    
    if valid_perps:
        avg_ppl = sum(valid_perps) / len(valid_perps)
        total_valid = len(valid_perps)
        logger.info(f"{model_name}: Average perplexity = {avg_ppl:.4f}, Valid samples = {total_valid}")
        return avg_ppl
    else:
        logger.warning(f"{model_name}: No valid perplexity values found")
        return None


def save_results(
    results: List[Optional[Dict[str, Any]]], 
    model_name: str, 
    dataset_file: str, 
    avg_ppl: Optional[float], 
    output_dir: str
) -> None:
    """Save results to JSON file."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate safe model name for filename
    if model_name == "ReasoningTransferability/UniReason-Qwen3-14B-RL":
        safe_model_name = "UniReason-RL"
    elif model_name == "ReasoningTransferability/UniReason-Qwen3-14B-think-SFT":
        safe_model_name = "UniReason-SFT"
    else:
        safe_model_name = model_name.split("/")[-1]
    
    dataset_basename = os.path.basename(dataset_file).replace(".jsonl", "")
    out_file = os.path.join(output_dir, f"{safe_model_name}_{dataset_basename}_token_losses.json")
    
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "model": model_name,
                "dataset": dataset_file,
                "results": results,
                "average_perplexity": avg_ppl
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {out_file}")
        
    except Exception as e:
        logger.error(f"Error saving results to {out_file}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch calculate token loss for each response in dataset using different models"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset JSON file path")
    parser.add_argument("--models", type=str, required=True,
                        help="Model name list, separated by commas")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Number of samples per mini-batch")
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="Output directory")
    
    args = parser.parse_args()



    # Parse model names
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    logger.info(f"Models to process: {model_names}")

    # Load dataset
    dataset = load_dataset(args.dataset)
    if not dataset:
        logger.error("No dataset loaded. Exiting.")
        return

    # Process each model
    all_results = {}
    avg_results = {}
    
    for model_name in model_names:
        results = process_model(model_name, dataset, args.batch_size)
        all_results[model_name] = results
        
        # Calculate and print statistics
        avg_ppl = calculate_and_print_stats(results, model_name)
        avg_results[model_name] = avg_ppl
        
        # Save results
        save_results(results, model_name, args.dataset, avg_ppl, args.output_dir)

    # Print summary
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    for model_name, avg_ppl in avg_results.items():
        total_valid = len([res for res in all_results[model_name] 
                          if res is not None and res.get("perplexity") is not None])
        print(f"{model_name}: Average perplexity = {avg_ppl:.4f}, "
              f"Valid samples = {total_valid}")


if __name__ == "__main__":
    main()
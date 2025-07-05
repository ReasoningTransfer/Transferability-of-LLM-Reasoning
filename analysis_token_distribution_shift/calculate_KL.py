# -*- coding: utf-8 -*-
"""
KL Divergence Calculator for Token-Level Analysis

This script calculates the KL divergence between different models' token-level predictions
from a generation model to target models. It processes JSON files containing token losses
and computes the divergence by averaging over tokens and then over data points.

Usage:
    python calculate_KL.py --data_dir <path_to_data> --target_gen_model <model_name>
"""

import os
import json
import logging
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import argparse


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_from_dir(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load all data files from the specified directory.
    
    Args:
        data_dir: Directory containing JSON files with token losses
        
    Returns:
        List of processed data entries
    """
    all_data = []
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return all_data
    
    json_files = [f for f in os.listdir(data_dir) 
                  if f.endswith('.json') and "token_losses" in f]
    
    logger.info(f"Found {len(json_files)} JSON files in {data_dir}")
    
    for filename in json_files:
        file_path = os.path.join(data_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract model information from data structure with proper mapping
                model_full_name = data['model']
                if model_full_name == "ReasoningTransferability/UniReason-Qwen3-14B-RL":
                    tested_model = "UniReason-RL"
                elif model_full_name == "ReasoningTransferability/UniReason-Qwen3-14B-think-SFT":
                    tested_model = "UniReason-SFT"
                else:
                    tested_model = model_full_name.split("/")[-1]
                
                dataset = data['dataset']
                generation_model = os.path.basename(dataset).split("_")[0]
                task = "_".join(os.path.basename(dataset).split("_")[1:]).replace(".jsonl", "")
                
                # Process results
                valid_results = 0
                for result in data['results']:
                    if result is not None and 'avg_loss' in result:
                        all_data.append({
                            'tested_model': tested_model,
                            'generation_model': generation_model,
                            'task': task,
                            'avg_loss': result['avg_loss'],
                            'perplexity': result['perplexity'],
                            'token_losses': result['token_losses']
                        })
                        valid_results += 1
                    else:
                        logger.warning(f"Invalid result found in file {filename}")
                
                logger.info(f"Loaded {valid_results} valid results from {filename}")
                
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
    
    logger.info(f"Total data entries loaded: {len(all_data)}")
    return all_data


def calculate_kl_divergence(data: List[Dict[str, Any]], target_gen_model: str) -> List[Dict[str, Any]]:
    """
    Calculate KL divergence from generation model to other models.
    
    For each token: P(token) / Q(token), take log, average over tokens, then average over data points.
    
    Args:
        data: List of data entries
        target_gen_model: Target generation model name
        
    Returns:
        List of KL divergence results
    """
    # Group data by generation model and task, storing each data point separately
    grouped_data = defaultdict(lambda: defaultdict(list))
    for entry in data:
        key = (entry['generation_model'], entry['task'])
        grouped_data[key][entry['tested_model']].append(entry['token_losses'])
    
    kl_results = []
    
    for (gen_model, task), models_data in grouped_data.items():
        logger.info(f"Processing: {gen_model} - {task}")
        
        # Only process data for the target generation model
        if gen_model != target_gen_model:
            continue
        
        model_names = list(models_data.keys())
        
        # Calculate KL divergence from generation model to other models
        for model in model_names:
            if gen_model != model:  # Only calculate KL to different models
                gen_data_points = models_data[gen_model]  # Generation model data points
                test_data_points = models_data[model]     # Test model data points
                
                logger.info(f"Generation model {gen_model} data points: {len(gen_data_points)}")
                logger.info(f"Test model {model} data points: {len(test_data_points)}")
                
                # Ensure both models have the same number of data points
                min_data_points = min(len(gen_data_points), len(test_data_points))
                if min_data_points == 0:
                    logger.warning(f"No data points found for {gen_model} or {model} in task {task}")
                    continue
                
                # Calculate KL divergence for each data point, then average over data points
                kl_per_data_point = []
                total_tokens = 0
                
                for i in range(min_data_points):
                    gen_token_losses = gen_data_points[i]    # Current data point token losses
                    test_token_losses = test_data_points[i]  # Current data point token losses
                    
                    # Ensure both models have the same number of tokens for the same data point
                    min_tokens = min(len(gen_token_losses), len(test_token_losses))
                    if min_tokens == 0:
                        continue
                    
                    # Calculate probability for each token (exp(-loss))
                    gen_probs = []
                    test_probs = []
                    
                    for j in range(min_tokens):
                        gen_loss = gen_token_losses[j]['loss']
                        test_loss = test_token_losses[j]['loss']
                        
                        # Convert loss to probability: P = exp(-loss)
                        gen_prob = np.exp(-gen_loss)
                        test_prob = np.exp(-test_loss)
                        
                        gen_probs.append(gen_prob)
                        test_probs.append(test_prob)
                    
                    # Add small smoothing term to avoid zero probabilities
                    epsilon = 1e-10
                    gen_probs = np.array(gen_probs) + epsilon
                    test_probs = np.array(test_probs) + epsilon
                    
                    # Calculate KL divergence for current data point: average over tokens
                    # KL(P||Q) = E[log(P/Q)] = E[log(P) - log(Q)]
                    log_ratio = np.log(gen_probs) - np.log(test_probs)
                    kl_current_data_point = np.mean(log_ratio)  # Average over tokens
                    
                    kl_per_data_point.append(kl_current_data_point)
                    total_tokens += min_tokens
                
                if len(kl_per_data_point) == 0:
                    logger.warning(f"No valid data points found for {gen_model} and {model} in task {task}")
                    continue
                
                # Average over data points
                final_kl = np.mean(kl_per_data_point)
                
                kl_results.append({
                    'generation_model': gen_model,
                    'task': task,
                    'tested_model': model,
                    'kl_divergence': abs(final_kl),  # Take absolute value
                    'num_data_points': len(kl_per_data_point),
                    'total_tokens': total_tokens,
                    'avg_tokens_per_data_point': total_tokens / len(kl_per_data_point) if len(kl_per_data_point) > 0 else 0
                })
    
    return kl_results


def print_results(kl_results: List[Dict[str, Any]]) -> None:
    """Print KL divergence results in a formatted way."""
    # Group results by generation model and task
    grouped_results = defaultdict(list)
    for result in kl_results:
        key = (result['generation_model'], result['task'])
        grouped_results[key].append(result)
    
    print("\nKL Divergence Results (from generation model to other models):")
    print("Calculation method: For each token, compute P/Q and take log, average over tokens, then average over data points")
    print("=" * 80)
    
    for (gen_model, task), results in grouped_results.items():
        print(f"\nGeneration Model: {gen_model}")
        print(f"Task: {task}")
        print("-" * 50)
        
        for result in results:
            print(f"From {result['generation_model']} to {result['tested_model']}: "
                  f"{result['kl_divergence']:.6f}")
            print(f"  Based on {result['num_data_points']} data points, "
                  f"average {result['avg_tokens_per_data_point']:.1f} tokens per data point")


def save_results(kl_results: List[Dict[str, Any]], output_file: str) -> None:
    """Save KL divergence results to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(kl_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Calculate KL divergence between models')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing data files')
    parser.add_argument('--target_gen_model', type=str, required=True,
                        help='Target generation model name')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path (default: kl_divergence_results.json in data_dir)')
    
    args = parser.parse_args()
    
    # Load data
    data = load_data_from_dir(args.data_dir)
    if not data:
        logger.error("No data loaded. Exiting.")
        return
    
    # Calculate KL divergence
    kl_results = calculate_kl_divergence(data, args.target_gen_model)
    
    if not kl_results:
        logger.warning("No KL divergence results calculated.")
        return
    
    # Print results
    print_results(kl_results)
    
    # Save results
    output_file = args.output_file or os.path.join(args.data_dir, 'kl_divergence_results.json')
    save_results(kl_results, output_file)


if __name__ == "__main__":
    main()
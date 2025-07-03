#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pca_shift_analysis.py
---------------------
This script computes and saves PCA shift analysis between a base model and a fine-tuned model.
It extracts hidden state features, computes principal components on the base model,
and measures the shift in average projections on these components for the fine-tuned model.
"""

import argparse  # for parsing command-line arguments
import os        # for file and directory operations
import random    # for random sampling
import torch     # for PyTorch model handling and tensor operations
import numpy as np  # for numerical computations
import pandas as pd  # for DataFrame creation and manipulation
import matplotlib as mpl  # for Matplotlib configuration
import matplotlib.pyplot as plt  # for plotting

from transformers import AutoModelForCausalLM, AutoTokenizer  # for loading pre-trained language models
from sklearn.decomposition import PCA  # for principal component analysis
from datasets import load_dataset  # for loading datasets from Hugging Face
import json   # for reading and writing JSON
import gc     # for garbage collection

# === Parse command-line arguments ===
parser = argparse.ArgumentParser(description="Compute PCA shift between two model checkpoints")
parser.add_argument('--base_model', type=str, required=True,
                    help="Name or path of the base (original) model checkpoint")
parser.add_argument('--fine_tuned_model', type=str, required=True,
                    help="Name or path of the fine-tuned or updated model checkpoint")
parser.add_argument('--task_type', type=str, required=True,
                    help="Task identifier for selecting dataset configuration")
parser.add_argument('--k', type=int, default=30,
                    help="Number of samples (queries) to draw from the dataset")
args = parser.parse_args()

# Assign parsed arguments to variables
checkpoint_before = args.base_model  # path to the original model
checkpoint_after = args.fine_tuned_model  # path to the updated model
data_type = args.task_type  # dataset/task identifier
k = args.k  # number of examples to sample

# === Set random seeds and device configuration ===
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# === Dataset configuration mapping ===
# Each entry defines the dataset path, split, and relevant columns
DATASET_CFG = {
    "AIME24":  dict(path="math-ai/aime24",            split="test",          cols=["problem", "solution"]),
    "AIME25":  dict(path="math-ai/aime25",            split="test",          cols=["problem", "answer"]),
    "MATH500": dict(path="HuggingFaceH4/MATH-500",    split="test",          cols=["problem", "solution", "answer"]),
    "GPQA-D":  dict(path="Idavidrein/gpqa",           subset="gpqa_diamond", split="train",             cols=[
        "Pre-Revision Question",
        "Pre-Revision Correct Answer",
        "Pre-Revision Incorrect Answer 1",
        "Pre-Revision Incorrect Answer 2",
        "Pre-Revision Incorrect Answer 3",
        "Pre-Revision Explanation",
    ]),               
    "Livecodebench": dict(path="livecodebench/code_generation_lite",
                          split="test",
                          cols=['question_content','public_test_cases'],
                          kwargs=dict(version_tag="release_v5")),
    "ACPBench": dict(path="ibm-research/acp_bench",   subset="acp_app_bool", split="test",
                     cols=["context", "question", "answer"]),
    "SimpleQA":  dict(path="basicv8vc/SimpleQA",      split="test",          cols=["problem", "answer"]),
    "IFEval":    dict(path="google/IFEval",           split="train",         cols=["prompt"]),
    "TruthfulQA":dict(path="EleutherAI/truthful_qa_mc", subset="multiple_choice",
                      split="validation",             cols=["question", "choices", "label"]),
    "HalluEval": dict(path="pminervini/HaluEval",     subset="qa",          split="data",
                     cols=["knowledge", "dialogue_history", "right_response", "hallucinated_response"]),
    "SuperGPQA_Hard": dict(path="m-a-p/SuperGPQA", split="train",
                     cols=["question","options", "answer"]),
    "ZebraLogicBench":  dict(path="allenai/ZebraLogicBench",    subset="mc_mode",  split="test",   
                     cols=["puzzle", "question","choices"]),
    "COQA": dict(path="stanfordnlp/coqa", split="validation",
                     cols=["story","questions", "answers"]),
    "Head_QA": dict(path="head_qa" ,split="test",
                     cols=['qtext', 'ra', 'answers']),
    "Mc_Taco": dict(path="mc_taco" ,split="test",
                     cols=['question', 'answer']),  
    "Sciq": dict(path="allenai/sciq", split="train",
                     cols=["question","correct_answer", "support"]),
    "Commonsense_qa": dict(path="tau/commonsense_qa", split="train",
                     cols=["questions","choices", "answerKey"]),
    "Mt-bench": dict(path="philschmid/mt-bench", split="train",
                     cols=["turns"]),
}

# === Function to load and sample text data ===
def load_text_samples(task_type: str, k: int = 30):
    """
    Load dataset for the given task and return k randomly sampled text examples.
    Joins specified columns into a single string per example.
    """
    cfg = DATASET_CFG[task_type]
    kwargs = cfg.get("kwargs", {})
    subset = cfg.get("subset", None)
    
    # Load dataset from Hugging Face
    if subset:
        ds = load_dataset(cfg["path"], subset, split=cfg["split"],
                          **kwargs, trust_remote_code=True)
    else:
        ds = load_dataset(cfg["path"], split=cfg["split"],
                          **kwargs, trust_remote_code=True)

    # Select only the columns that exist in the dataset
    cols = [c for c in cfg["cols"] if c in ds.column_names]
    texts = ["  ".join(str(example[c]) for c in cols if example[c] is not None)
             for example in ds]

    random.shuffle(texts)
    return texts[:k]

# Special handling for certain task types
if data_type == "Olympia":
    # Read from a local JSONL file for the "Olympia" task
    file_path = "dataset/olympiadbench.jsonl"
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    sampled = random.sample(data, k)
    texts = [f"{item.get('question', '')} {item.get('solution', '')} {item.get('final_answer', '')}"
             for item in sampled]
elif data_type == "Mt-bench":
    # Use a larger sample size for Mt-bench
    texts = load_text_samples(data_type, k=80)
else:
    texts = load_text_samples(data_type, k=k)

# === Function to extract mean hidden-state features ===
def extract_features(model, tokenizer, texts, layer_idx):
    """
    Tokenize input texts and extract mean pooled hidden states from a specific layer.
    Returns a NumPy array of shape (num_examples, hidden_size).
    """
    inputs = tokenizer(texts, return_tensors='pt', padding=True,
                       truncation=True, max_length=128).to(model.device)
    outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states[layer_idx].to(torch.float32)
    return hs.mean(dim=1).detach().cpu().numpy()

# Load tokenizer and ensure pad token is defined
tokenizer = AutoTokenizer.from_pretrained(checkpoint_before, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# === Function to load a model checkpoint ===
def load_model(path):
    """
    Load a causal LM model in bfloat16 and map it to available devices with flash attention.
    """
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attention_2=True,
        device_map="auto"
    )
    return model.eval()

# === Function to compute centroid L2 distances ===
def compute_centroid_distance(df, states=["step1"]):
    """
    Compute L2 distance between the centroid of 'Original' state and each specified state.
    """
    orig = df[df["state"] == "Original"][['shift', 'principle']].mean().values
    result = {}
    for state in states:
        if state in df["state"].unique():
            other = df[df["state"] == state][['shift', 'principle']].mean().values
            result[state] = np.linalg.norm(other - orig)
    return result

# === Phase 1: Extract PCA components from the base model ===
model_before = load_model(checkpoint_before)
num_layers = model_before.config.num_hidden_layers + 1
orig_stats = []  # to store principal components and means per layer

for layer in range(num_layers):
    # Extract features for the original model
    with torch.no_grad():
        feats_o = extract_features(model_before, tokenizer, texts, layer)

    # Perform PCA on CPU
    pca = PCA(n_components=2).fit(feats_o)
    comp1, comp2 = pca.components_[0], pca.components_[1]
    pc1_o = float(feats_o.dot(comp1).mean())
    pc2_o = float(feats_o.dot(comp2).mean())
    orig_stats.append((comp1, comp2, pc1_o, pc2_o))

    # Clean up to free memory
    del feats_o
    gc.collect()

# Clean up the base model
del model_before
torch.cuda.empty_cache()
gc.collect()

# === Phase 2: Compute shifts on the updated model ===
model_un = load_model(checkpoint_after)
records = []  # list to hold shift and principle values for each layer/state

for layer, (comp1, comp2, pc1_o, pc2_o) in enumerate(orig_stats):
    with torch.no_grad():
        feats_u = extract_features(model_un, tokenizer, texts, layer)

    # Compute new projections
    pc1_u = float(feats_u.dot(comp1).mean())
    pc2_u = float(feats_u.dot(comp2).mean())
    shift = pc1_u - pc1_o

    # Record original (no shift) and updated states
    records.append({"layer": layer, "state": "Original", "shift": 0.0, "principle": pc2_o})
    records.append({"layer": layer, "state": "Updated",  "shift": shift, "principle": pc2_u})

    # Free up memory
    del feats_u
    gc.collect()

# Clean up the updated model
del model_un
torch.cuda.empty_cache()
gc.collect()

# Create a DataFrame from the records
df = pd.DataFrame(records)

# Prepare JSON entry for output
new_entry = {
    "benchmark": f"{data_type}_PCA_Shift",
    "data": records
}

# Determine output file path based on checkpoint name
basename = os.path.basename(checkpoint_after)
output_path = f"{basename}_pca_shift.json"

# Append to existing data if file exists
if os.path.exists(output_path):
    with open(output_path, "r") as f:
        existing_data = json.load(f)
    if isinstance(existing_data, list):
        existing_data.append(new_entry)
    else:
        existing_data = [existing_data, new_entry]
else:
    existing_data = [new_entry]

# Write final JSON data
with open(output_path, "w") as f:
    json.dump(existing_data, f, indent=2)

# Compute centroid distances
result_distance = compute_centroid_distance(df)

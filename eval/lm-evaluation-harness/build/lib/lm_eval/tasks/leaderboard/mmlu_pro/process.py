import pandas as pd
import numpy as np

def sample_parquet(input_file, output_file, sample_size=1000):
    # 读取parquet文件
    df = pd.read_parquet(input_file)
    
    # 检查是否有subject列
    if 'category' not in df.columns:
        raise ValueError("The dataset must contain a 'category' column.")
    
    # 计算每个subject的比例
    category_counts = df['category'].value_counts(normalize=True)
    
    # 计算每个subject应该抽取的数据量
    sampled_dfs = []
    for category, proportion in category_counts.items():
        category_df = df[df['category'] == category]
        n_samples = max(1, int(proportion * sample_size))  # 至少取1个样本
        sampled_dfs.append(category_df.sample(n=min(n_samples, len(category_df)), random_state=42))
    
    sampled_df = pd.concat(sampled_dfs).sample(n=sample_size, random_state=42, replace=True)
    
    # 存储为新的Parquet文件
    sampled_df.to_parquet(output_file, index=False)
    print(f"Sampled dataset saved to {output_file}")

# 使用示例
sample_parquet("/map-vepfs/ziyu/reasoning_tr/lm-evaluation-math/lm_eval/tasks/leaderboard/mmlu_pro/test-00000-of-00001.parquet", "/map-vepfs/ziyu/reasoning_tr/lm-evaluation-math/lm_eval/tasks/leaderboard/mmlu_pro/1k_test.parquet")

import pandas as pd
import os

def main():
    human_path = "human_answers.csv"
    v2_path = "../data /clean_data_with_features_v2.csv"
    if not os.path.exists(v2_path):
        v2_path = "../data/clean_data_with_features_v2.csv"
    if not os.path.exists(v2_path):
        v2_path = "./data /clean_data_with_features_v2.csv"
    if not os.path.exists(v2_path):
        v2_path = "./data/clean_data_with_features_v2.csv"

    if not os.path.exists(human_path):
        print("Error: 'human_answers.csv' not found. Please complete Task 1.")
        return

    if not os.path.exists(v2_path):
        print("Error: LLM feature extraction output file not found yet.")
        return

    df_human = pd.read_csv(human_path).set_index("row_idx")
    df_v2 = pd.read_csv(v2_path)

    q_cols = [f"q{i}" for i in range(1, 8)]
    
    # Check if human answers are populated
    if df_human[q_cols].isna().all().all():
        print("Warning: 'human_answers.csv' is empty. Fill in your answers first.")
        return

    sample_indices = df_human.index.tolist()
    df_model_sample = df_v2.loc[sample_indices, q_cols]

    print("=== HUMAN vs. LLM MODEL AGREEMENT EVALUATION ===\n")
    
    agreements = {}
    total_matches = 0
    total_comparisons = 0

    for q in q_cols:
        matches = (df_human[q].str.strip().str.lower() == df_model_sample[q].astype(str).str.strip().str.lower()).sum()
        total = len(sample_indices)
        pct = (matches / total) * 100
        agreements[q] = pct
        total_matches += matches
        total_comparisons += total
        print(f"{q} Agreement: {matches}/{total} ({pct:.1f}%)")

    overall_pct = (total_matches / total_comparisons) * 100
    print("-" * 50)
    print(f"Overall Model-vs-Human Agreement: {overall_pct:.1f}%\n")

if __name__ == "__main__":
    main()

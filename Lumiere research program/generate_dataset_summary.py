import os
import pandas as pd
import numpy as np

# Locate dataset file
path_candidates = [
    "./data /clean_data_with_features_v2.csv",
    "./data/clean_data_with_features_v2.csv",
    "clean_data_with_features_v2.csv",
    "../data /clean_data_with_features_v2.csv",
    "../data/clean_data_with_features_v2.csv"
]
path = next((p for p in path_candidates if os.path.exists(p)), None)

if not path:
    print("Error: Could not find 'clean_data_with_features_v2.csv'. Please check your path.")
else:
    df = pd.read_csv(path)
    
    print("=" * 65)
    print("      DATASET SUMMARY STATISTICS REPORT (clean_data_with_features_v2)")
    print("=" * 65)
    print(f"Total Rows (Observations): {len(df)}")
    print(f"Total Columns: {len(df.columns)}\n")

    # 1. Unique Countries
    country_cols = [c for c in df.columns if 'country' in c.lower()]
    if country_cols:
        col_name = country_cols[0]
        n_countries = df[col_name].nunique(dropna=True)
        print(f"1. Unique Countries: {n_countries} (from column '{col_name}')")
    else:
        print("1. Unique Countries: N/A (No 'country' column present in this dataset file)")

    # 2. Unique Field Partners
    if 'field_partner_id' in df.columns:
        n_fp = df['field_partner_id'].nunique(dropna=True)
        print(f"2. Unique Field Partners (field_partner_id): {n_fp}")
    else:
        print("2. Unique Field Partners: Column 'field_partner_id' not found")

    # 3. Average Description Word Length
    if 'description' in df.columns:
        word_lengths = df['description'].fillna('').apply(lambda x: len(str(x).split()))
        avg_words = word_lengths.mean()
        print(f"3. Average Loan Description Length: {avg_words:.2f} words per description")
    else:
        print("3. Average Description Length: Column 'description' not found")

    # 4. Mean & Standard Deviation of Key Numeric Variables
    num_cols = ['loan_amount', 'lender_repayment_term_months', 'borrower_count', 'funding_speed_days']
    print("\n4. Key Numeric Features (Mean & Standard Deviation):")
    for col in num_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"   - {col:<32}: Mean = {mean_val:10.2f} | Std Dev = {std_val:10.2f}")
        else:
            print(f"   - {col:<32}: Column not found")

    # 5. Distribution of sector_group
    print("\n5. Distribution of sector_group:")
    if 'sector_group' in df.columns:
        sector_counts = df['sector_group'].value_counts(dropna=False)
        sector_pcts = df['sector_group'].value_counts(normalize=True, dropna=False) * 100
        for cat in sector_counts.index:
            cnt = sector_counts[cat]
            pct = sector_pcts[cat]
            print(f"   - {str(cat):<28}: {cnt:>5} rows ({pct:6.2f}%)")
    else:
        print("   - Column 'sector_group' not found")

    # 6. Binary Label Distribution
    print("\n6. Binary Target Label Distribution:")
    if 'label' in df.columns:
        label_counts = df['label'].value_counts(dropna=False)
        label_pcts = df['label'].value_counts(normalize=True, dropna=False) * 100
        rep_cnt, rep_pct = label_counts.get(0, 0), label_pcts.get(0, 0.0)
        def_cnt, def_pct = label_counts.get(1, 0), label_pcts.get(1, 0.0)
        print(f"   - Label = 0 (Repaid)  : {rep_cnt:>5} rows ({rep_pct:6.2f}%)")
        print(f"   - Label = 1 (Default) : {def_cnt:>5} rows ({def_pct:6.2f}%)")
    else:
        print("   - Column 'label' not found")

    # 7. List of Retained Column Names
    print("\n7. Retained Columns in Model-Ready Dataset:")
    for idx, col in enumerate(df.columns, 1):
        print(f"   {idx:2d}. {col}")
    print("=" * 65)

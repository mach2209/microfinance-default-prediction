import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Locate Dataset File
path_candidates = [
    "./data /clean_data_with_features_v2.csv",
    "./data/clean_data_with_features_v2.csv",
    "clean_data_with_features_v2.csv",
    "../data/clean_data_with_features_v2.csv"
]
path = next((p for p in path_candidates if os.path.exists(p)), None)

if not path:
    print("Error: Could not find 'clean_data_with_features_v2.csv'.")
    exit(1)

df = pd.read_csv(path)

# 2. Define Features
baseline_num_cols = ['loan_amount', 'lender_repayment_term_months', 'borrower_count', 'raised_year', 'funding_speed_days', 'pct_pictured']
baseline_cat_cols = ['gender', 'group_or_individual', 'sector_group', 'field_partner_id']
llm_features = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']

# Ensure only existing columns are used
baseline_num_cols = [c for c in baseline_num_cols if c in df.columns]
baseline_cat_cols = [c for c in baseline_cat_cols if c in df.columns]
llm_features = [c for c in llm_features if c in df.columns]

# 3. Encode LLM Features (Yes=1.0, No=0.0, Difficult to tell=0.5)
def map_q_responses(val):
    val_str = str(val).strip().lower()
    if 'yes' in val_str:
        return 1.0
    elif 'no' in val_str:
        return 0.0
    else:
        return 0.5  # Captures "Difficult to tell" or any missing values

for col in llm_features:
    df[col] = df[col].apply(map_q_responses)

# Full numerical columns list includes mapped LLM features
full_num_cols = baseline_num_cols + llm_features

# 4. Build Preprocessing Pipelines
def build_pipeline(num_features, cat_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ]
    )
    
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    return clf

clf_baseline = build_pipeline(baseline_num_cols, baseline_cat_cols)
clf_full = build_pipeline(full_num_cols, baseline_cat_cols)

# 5. Define Data & Cross-Validation
X_baseline = df[baseline_num_cols + baseline_cat_cols]
X_full = df[full_num_cols + baseline_cat_cols]
y = df['label']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline_aucs = []
full_aucs = []

print("============================================================")
print("      MODEL COMPARISON: BASELINE VS. LLM-AUGMENTED FULL")
print("============================================================")

# 6. Execute Fold Evaluation
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train & Evaluate Baseline
    X_base_train, X_base_val = X_baseline.iloc[train_idx], X_baseline.iloc[val_idx]
    clf_baseline.fit(X_base_train, y_train)
    y_base_proba = clf_baseline.predict_proba(X_base_val)[:, 1]
    base_auc = roc_auc_score(y_val, y_base_proba)
    baseline_aucs.append(base_auc)
    
    # Train & Evaluate Full
    X_full_train, X_full_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
    clf_full.fit(X_full_train, y_train)
    y_full_proba = clf_full.predict_proba(X_full_val)[:, 1]
    full_auc = roc_auc_score(y_val, y_full_proba)
    full_aucs.append(full_auc)
    
    print(f"Fold {fold}: Baseline AUC = {base_auc:.4f} | Full AUC = {full_auc:.4f} | Diff = {full_auc - base_auc:+.4f}")

# 7. Compute Summary Metrics
mean_base = np.mean(baseline_aucs)
std_base = np.std(baseline_aucs)

mean_full = np.mean(full_aucs)
std_full = np.std(full_aucs)

print("-" * 60)
print(f"Baseline Mean AUC : {mean_base:.4f} (Std: {std_base:.4f})")
print(f"Full Model Mean AUC: {mean_full:.4f} (Std: {std_full:.4f})")
print(f"Absolute AUC Delta : {mean_full - mean_base:+.4f}")
print("============================================================")

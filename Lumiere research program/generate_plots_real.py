import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set clean professional plot styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

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

# 2. Define Exact Feature Sets
baseline_num_cols = ['loan_amount', 'lender_repayment_term_months', 'borrower_count', 'raised_year', 'funding_speed_days', 'pct_pictured']
baseline_cat_cols = ['gender', 'group_or_individual', 'sector_group', 'field_partner_id']
llm_features = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']

# 3. Map LLM Features (Yes=1.0, No=0.0, Difficult to tell=0.5)
def map_q_responses(val):
    val_str = str(val).strip().lower()
    if 'yes' in val_str:
        return 1.0
    elif 'no' in val_str:
        return 0.0
    else:
        return 0.5

for col in llm_features:
    df[col] = df[col].apply(map_q_responses)

full_num_cols = baseline_num_cols + llm_features

# 4. Construct Pipelines
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

X_baseline = df[baseline_num_cols + baseline_cat_cols]
X_full = df[full_num_cols + baseline_cat_cols]
y = df['label']

# 5. Run 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline_aucs = []
full_aucs = []

fold5_y_val = None
fold5_base_proba = None
fold5_full_proba = None

print("Fitting cross-validation folds on real dataset...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_baseline, y), 1):
    X_base_tr, X_base_val = X_baseline.iloc[train_idx], X_baseline.iloc[val_idx]
    X_full_tr, X_full_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Baseline Model
    clf_baseline.fit(X_base_tr, y_tr)
    p_base = clf_baseline.predict_proba(X_base_val)[:, 1]
    auc_base = roc_auc_score(y_val, p_base)
    baseline_aucs.append(auc_base)
    
    # Full Model
    clf_full.fit(X_full_tr, y_tr)
    p_full = clf_full.predict_proba(X_full_val)[:, 1]
    auc_full = roc_auc_score(y_val, p_full)
    full_aucs.append(auc_full)
    
    # Store Fold 5 data dynamically
    if fold == 5:
        fold5_y_val = y_val
        fold5_base_proba = p_base
        fold5_full_proba = p_full

# =====================================================================
# PLOT 1: ROC CURVE COMPARISON (FOLD 5 ONLY)
# =====================================================================
fpr_base, tpr_base, _ = roc_curve(fold5_y_val, fold5_base_proba)
auc_f5_base = roc_auc_score(fold5_y_val, fold5_base_proba)

fpr_full, tpr_full, _ = roc_curve(fold5_y_val, fold5_full_proba)
auc_f5_full = roc_auc_score(fold5_y_val, fold5_full_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr_base, tpr_base, label=f"Baseline Model (AUC = {auc_f5_base:.4f})", color='#1f77b4', linewidth=2)
plt.plot(fpr_full, tpr_full, label=f"Full Model (AUC = {auc_f5_full:.4f})", color='#ff7f0e', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title("ROC Curve: Baseline vs. LLM-Augmented Model", fontweight='bold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=300)
plt.close()

print(" -> Saved roc_curve_comparison.png")

# =====================================================================
# PLOT 2: AUC BY FOLD (SIDE-BY-SIDE BARS)
# =====================================================================
folds = np.arange(1, 6)
width = 0.35
mean_base = np.mean(baseline_aucs)
mean_full = np.mean(full_aucs)

plt.figure(figsize=(8, 6))
plt.bar(folds - width/2, baseline_aucs, width, label='Baseline', color='#1f77b4')
plt.bar(folds + width/2, full_aucs, width, label='Full Model', color='#ff7f0e')

plt.axhline(mean_base, color='#1f77b4', linestyle='--', linewidth=1.5, label=f'Baseline Mean: {mean_base:.4f}')
plt.axhline(mean_full, color='#ff7f0e', linestyle='--', linewidth=1.5, label=f'Full Model Mean: {mean_full:.4f}')

plt.ylim(0.8, 1.0)
plt.xlabel("Cross-Validation Fold")
plt.ylabel("Area Under ROC Curve (AUC)")
plt.title("AUC per Fold: Baseline vs. Full Model", fontweight='bold')
plt.xticks(folds, [f"Fold {i}" for i in folds])
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("auc_by_fold.png", dpi=300)
plt.close()

print(" -> Saved auc_by_fold.png")

# =====================================================================
# PLOT 3: FEATURE COEFFICIENTS (FULL MODEL ON ENTIRE DATASET)
# =====================================================================
# Train full model on complete dataset
clf_full.fit(X_full, y)
coefs = clf_full.named_steps['classifier'].coef_[0]

# Numerical features are mapped first in ColumnTransformer
q_indices = [full_num_cols.index(q) for q in llm_features]
q_coefs = [coefs[i] for i in q_indices]

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#d62728' if c > 0 else '#1f77b4' for c in q_coefs]
y_pos = np.arange(len(llm_features))

ax.barh(y_pos, q_coefs, color=colors, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels([q.upper() for q in llm_features])
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel("Coefficient (Log-Odds)")
ax.set_title("LLM Feature Coefficients (Log-Odds)", fontweight='bold')

# Scale x limits dynamically to avoid text clipping
max_abs_coef = max(abs(min(q_coefs)), abs(max(q_coefs)))
ax.set_xlim(-max_abs_coef * 1.4, max_abs_coef * 1.4)

for i, coef in enumerate(q_coefs):
    odds_ratio = np.exp(coef)
    ha = 'left' if coef >= 0 else 'right'
    x_offset = max_abs_coef * 0.04 if coef >= 0 else -max_abs_coef * 0.04
    ax.text(coef + x_offset, i, f"OR: {odds_ratio:.2f}", va='center', ha=ha, fontsize=10, fontweight='bold')

ax.invert_yaxis()
plt.tight_layout()
plt.savefig("feature_coefficients.png", dpi=300)
plt.close()

print(" -> Saved feature_coefficients.png")
print("\nAll 3 figures successfully generated from clean_data_with_features_v2.csv.")

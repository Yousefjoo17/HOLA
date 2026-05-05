import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("============================================================")
print(" PHASE 6: Predictive Modeling (Random Forest) - GLOBAL METRICS")
print("============================================================\n")

# 1. Load the optimized dataset
print("Loading final_customer_profile_optimized.csv...")
try:
    df = pd.read_csv("final_customer_profile_optimized.csv")
except FileNotFoundError:
    print("Error: 'final_customer_profile_optimized.csv' not found.")
    exit()

# 2. Identify Features (X) and Targets (Y)
target_cols = [col for col in df.columns if col.startswith('HAS_PROD_')]
exclude_cols = ['CUSTOMER_ID', 'BRANCH_ID'] + target_cols
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].fillna(0)

# Filter targets to only those with enough samples
MIN_SAMPLES_REQUIRED = 50
valid_targets = [col for col in target_cols if df[col].sum() >= MIN_SAMPLES_REQUIRED]

print(f"Total Features used for modeling: {len(feature_cols)}")
print(f"Total Valid Products to evaluate: {len(valid_targets)}\n")

# Prepare Target Matrix (Y) with only the valid products
Y = df[valid_targets]

# 3. GLOBAL Train-Test Split
# We split once so that X_test contains the exact same customers for every product
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Dictionary to collect predictions for all products
predictions_dict = {}

# 4. Train a Random Forest Model for each valid Product
print("Training models for each product...")
for product in valid_targets:
    y_train = Y_train[product]
    
    # Train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Store predictions
    predictions_dict[product] = rf_model.predict(X_test)

# Convert predictions dictionary into a DataFrame that mirrors the shape of Y_test
Y_pred = pd.DataFrame(predictions_dict, index=Y_test.index)

# ============================================================
# 5. Global Metric Calculation
# ============================================================
print("\n============================================================")
print(" GLOBAL EVALUATION METRICS (Across All Products)")
print("============================================================\n")

# Micro Average: Counts total true positives, false negatives, and false positives globally
micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
    Y_test, Y_pred, average='micro', zero_division=0
)

# Macro Average: Calculates metrics for each product separately, then averages them equally
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    Y_test, Y_pred, average='macro', zero_division=0
)

print("--- MICRO AVERAGE ---")
print("Best for overall performance evaluation across the dataset.")
print(f"Precision: {micro_precision:.4f}")
print(f"Recall:    {micro_recall:.4f}")
print(f"F1-Score:  {micro_f1:.4f}\n")

print("--- MACRO AVERAGE ---")
print("Best for evaluating performance evenly across all products, regardless of class imbalance.")
print(f"Precision: {macro_precision:.4f}")
print(f"Recall:    {macro_recall:.4f}")
print(f"F1-Score:  {macro_f1:.4f}\n")
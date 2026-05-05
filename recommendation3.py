import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("============================================================")
print(" PHASE 6: Predictive Modeling (Random Forest)")
print("============================================================\n")

# 1. Load the optimized dataset
print("Loading final_customer_profile_optimized.csv...")
try:
    df = pd.read_csv("final_customer_profile_optimized.csv")
except FileNotFoundError:
    print("Error: 'final_customer_profile_optimized.csv' not found. Please ensure the previous phases completed successfully.")
    exit()

# 2. Identify Features (X) and Targets (Y)
target_cols = [col for col in df.columns if col.startswith('HAS_PROD_')]
exclude_cols = ['CUSTOMER_ID', 'BRANCH_ID'] + target_cols
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Prepare Feature Matrix (X) - Filling any edge-case NaNs with 0
X = df[feature_cols].fillna(0)

print(f"Total Features used for modeling: {len(feature_cols)}")
print(f"Total Target Products to evaluate: {len(target_cols)}\n")

# 3. Train a Random Forest Model for each Product
# We will only evaluate products that have at least 50 positive samples to ensure valid train/test splits
MIN_SAMPLES_REQUIRED = 50

for product in target_cols:
    y_prod = df[product]
    
    # Check if we have enough positive samples to train a meaningful model
    positive_cases = y_prod.sum()
    if positive_cases < MIN_SAMPLES_REQUIRED:
        print(f"[Skipping] {product.replace('HAS_PROD_', '')}: Only {positive_cases} customers have this (requires {MIN_SAMPLES_REQUIRED}).")
        continue

    print(f"\n{'='*60}")
    print(f" TARGET: {product.replace('HAS_PROD_', '')} (Total Owners: {positive_cases})")
    print(f"{'='*60}")
    
    # 4. Train-Test Split (80% Training, 20% Testing)
    # We use stratify=y_prod to ensure the 80/20 split maintains the same ratio of product owners vs non-owners
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_prod, test_size=0.2, random_state=42, stratify=y_prod
    )
    
    # 5. Initialize and Train Random Forest
    # class_weight='balanced' is crucial here because most customers will NOT have the product (class imbalance)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # 6. Predict on unseen Test Data
    y_pred = rf_model.predict(X_test)
    
    # 7. Evaluate Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    
    print("Classification Report:")
    # The classification report shows Precision, Recall, and F1-Score for both classes (0 = No Product, 1 = Has Product)
    print(classification_report(y_test, y_pred))
    
    # 8. Extract Top 5 Most Important Features for this specific product
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Top 5 Driving Features for this Product:")
    for feature, importance in importances.head(5).items():
        print(f"  - {feature}: {importance:.4f}")

print("\nModel evaluation complete!")
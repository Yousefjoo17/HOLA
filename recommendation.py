from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

print("\n========================= Product Recommendation System =========================\n")

# ---------------------------------------------------------
# 1. Feature Selection & Target Isolation
# ---------------------------------------------------------
# Identify our target variables (the product holdings we want to predict)
target_cols = [col for col in prime_df.columns if col.startswith('HAS_PROD_')]
y = prime_df[target_cols]

# For features (X), we only want numeric/scaled columns. 
# We MUST drop the target columns, and any unique identifiers (like RIMNO, BRANCH_ID) 
# to prevent data leakage and overfitting.
X = prime_df.select_dtypes(include=[np.number]).drop(columns=target_cols + ['RIMNO', 'BRANCH_ID'], errors='ignore')

# ---------------------------------------------------------
# 2. Train / Test Split
# ---------------------------------------------------------
# Standard 80/20 split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape:  {X_test.shape}")

# ---------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------
print("\nTraining Random Forest Recommendation Model...")
# n_jobs=-1 uses all available CPU cores to speed up training
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. Evaluation & Accuracy
# ---------------------------------------------------------
y_pred = clf.predict(X_test)

# Exact Match Accuracy: In multi-label classification, this checks if the 
# predicted combination of products is a 100% exact match to the actuals.
exact_match_accuracy = accuracy_score(y_test, y_pred)
print(f"\nExact Match Accuracy: {exact_match_accuracy * 100:.2f}%")

# Because Exact Match is very strict, it's highly recommended to look at precision/recall per product:
print("\nClassification Report (per product):")
print(classification_report(y_test, y_pred, target_names=target_cols, zero_division=0))


# ---------------------------------------------------------
# 5. The Recommendation Engine
# ---------------------------------------------------------
def recommend_top_n_products(model, customer_features, product_names, top_n=2):
    """
    Takes customer features and predicts the top N recommended products 
    based on the model's confidence probabilities.
    """
    # predict_proba returns a list of arrays for multi-label (one array per target class)
    proba_list = model.predict_proba(customer_features)
    
    # Extract the probability of the positive class (class 1) for each product
    # proba_list[i][:, 1] gets the probability that the target is 1
    positive_probs = np.array([prob[:, 1] if prob.shape[1] > 1 else prob[:, 0] * 0 for prob in proba_list]).T
    
    recommendations = []
    for i in range(len(customer_features)):
        # Get indices of the highest probabilities for this specific user
        top_indices = np.argsort(positive_probs[i])[::-1][:top_n]
        
        # Map indices back to clean product names and pair with their probability scores
        user_recs = [
            (product_names[idx].replace('HAS_PROD_', ''), positive_probs[i][idx]) 
            for idx in top_indices
        ]
        recommendations.append(user_recs)
        
    return recommendations

# --- Test the Recommender on 5 Sample Customers ---
print("\n--- Sample Recommendations for Top 5 Test Customers ---")
sample_customers = X_test.head(5)
recs = recommend_top_n_products(clf, sample_customers, target_cols, top_n=3)

# Display the recommendations cleanly
for i, (idx, rec) in enumerate(zip(sample_customers.index, recs)):
    # If you mapped RIMNO back to the index, you can print the RIMNO here
    print(f"Customer Profile Index [{idx}]:")
    for rank, (product, prob) in enumerate(rec, 1):
        print(f"  {rank}. {product} (Match Score: {prob*100:.1f}%)")
        

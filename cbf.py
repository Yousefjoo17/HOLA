import pandas as pd 
import numpy as np
import glob
from sklearn.metrics.pairwise import cosine_similarity

# ========================= 1. Load Data ==========================
prime_files = glob.glob("/path/to/prime/*.csv")
prime_df_list = []

print("Loading CSV files...")
for file in prime_files:
    temp_df = pd.read_csv(
        file, 
        encoding='latin'
    ).rename(columns={'RIM_NO': 'RIMNO', "NAME": "PRODUCT_NAME"})
    
    prime_df_list.append(temp_df)
    
prime_df = pd.concat(prime_df_list, ignore_index=True)


def drop_empty_records(df, columns):
    """
    Drops rows from a dataframe if they are missing data in the specified columns.
    Prints a summary of how many rows were removed.
    """
    # Safeguard: if a single string is passed instead of a list, convert it
    if isinstance(columns, str):
        columns = [columns]
        
    # Filter the list to only include columns that actually exist in the dataframe
    valid_cols = [col for col in columns if col in df.columns]
    
    if not valid_cols:
        print(f"\nWarning: None of the specified columns {columns} exist in the dataframe. No rows dropped.")
        return df
        
    # Count rows before dropping
    rows_before = len(df)
    
    # Drop the empty records
    cleaned_df = df.dropna(subset=valid_cols, how='any')
    
    # Report the results
    rows_after = len(cleaned_df)
    rows_dropped = rows_before - rows_after
    
    print(f"\n--- Dropping Empty Records ---")
    print(f"Checked columns: {valid_cols}")
    print(f"Dropped {rows_dropped} rows.")
    print(f"Total rows remaining: {rows_after}")
    
    return cleaned_df

critical_columns = ['RIMNO']
prime_df = drop_empty_records(prime_df, critical_columns)

Product_frequency_threshold = 500

# Group rare products
prime_counts = prime_df["PRODUCT_NAME"].value_counts()
prime_df["PRODUCT_NAME"] = prime_df["PRODUCT_NAME"].map(
    lambda x: x if prime_counts.get(x, 0) >= Product_frequency_threshold else "another product"
)

# Create a binary matrix of Customer (RIMNO) vs. Product Holding
user_item_matrix = pd.crosstab(prime_df['RIMNO'], prime_df['PRODUCT_NAME'])

# Prefix the columns so they don't get confused with future features
user_item_matrix.columns = [f"HAS_PROD_{col.strip().replace(' ', '_')}" for col in user_item_matrix.columns]

# Reset the index to make RIMNO a regular column, and overwrite prime_df
# This naturally drops all the original columns (BRANCH_NAME, CREDIT_LIMIT, etc.)
prime_df = user_item_matrix.reset_index()

# Convert values to strictly 1s and 0s if a customer can have multiple of the same product
prime_df.iloc[:, 1:] = (prime_df.iloc[:, 1:] > 0).astype(int)

print("\n--- Final Output Dataframe ---")
print(prime_df.head())
print(prime_df.info())


# ========================= 2. Build Collaborative Filtering Engine =========================
print("\n--- Building Collaborative Filtering Engine ---")

# 1. Set RIMNO as the index so our matrix contains ONLY product 1s and 0s
user_item_matrix = prime_df.set_index('RIMNO')

# 2. Calculate Item-Item Cosine Similarity
# We transpose (.T) the matrix because we want similarity between columns (products), not rows (users)
item_sim_array = cosine_similarity(user_item_matrix.T)

# 3. Convert the array back into a readable DataFrame
item_sim_df = pd.DataFrame(
    item_sim_array, 
    index=user_item_matrix.columns, 
    columns=user_item_matrix.columns
)

print("Item Similarity Matrix created successfully.")


# ========================= 3. Train/Test Split (Masking) ==========================
def create_train_test_split(user_matrix, test_ratio, random_state=42):
    """
    Splits the user-item matrix by masking a percentage of known positives (1s).
    Users with only 1 product are kept entirely in the train set.
    """
    np.random.seed(random_state)
    
    # Initialize train as a copy, and test as all zeros
    train_matrix = user_matrix.copy()
    test_matrix = pd.DataFrame(0, index=user_matrix.index, columns=user_matrix.columns)
    
    users_with_test_items = 0
    
    # Iterate over the user index locations
    for user_idx in range(user_matrix.shape[0]):
        # Get indices of columns where the user holds a product (value > 0)
        held_item_indices = np.where(user_matrix.iloc[user_idx].values > 0)[0]
        
        # We can only mask items if the user holds more than 1 product
        if len(held_item_indices) > 1:
            # Determine how many items to mask
            num_test_items = max(1, int(len(held_item_indices) * test_ratio))
            
            # Randomly select items to move to the test set
            test_indices = np.random.choice(held_item_indices, size=num_test_items, replace=False)
            
            # Move selected items to test matrix, remove from train matrix
            test_matrix.iloc[user_idx, test_indices] = 1
            train_matrix.iloc[user_idx, test_indices] = 0
            
            users_with_test_items += 1
            
    print(f"Created split. {users_with_test_items} out of {user_matrix.shape[0]} users have items in the test set.")
    return train_matrix, test_matrix

# Execute the split
train_df, test_df = create_train_test_split(user_item_matrix, test_ratio=0.2)

# ========================= 2. Re-train Similarity on Train Data =====================
# Important: We must calculate item similarity ONLY on the training data 
# to prevent data leakage from the test set.
train_item_sim_array = cosine_similarity(train_df.T)
train_item_sim_df = pd.DataFrame(
    train_item_sim_array, 
    index=train_df.columns, 
    columns=train_df.columns
)


# ========================= 3. Evaluation Metrics ====================================
def get_top_k_recommendations(user_row, sim_matrix, k=3):
    """Helper function to get top K recommendations for a single user row."""
    held_products = user_row[user_row > 0].index.tolist()
    
    if not held_products:
        return []
        
    # Calculate scores based on held products
    scores = sim_matrix[held_products].sum(axis=1)
    # Drop items already held in the train set so we don't recommend them
    scores = scores.drop(held_products)
    
    # Return top K product names
    return scores.sort_values(ascending=False).head(k).index.tolist()


def evaluate_model(train_matrix, test_matrix, sim_matrix, k=3):
    """
    Calculates Precision@K and Recall@K across all valid users.
    """
    precisions = []
    recalls = []
    
    for user_idx in range(train_matrix.shape[0]):
        # Extract the user's hidden test items
        test_row = test_matrix.iloc[user_idx]
        actual_test_items = test_row[test_row > 0].index.tolist()
        
        # If this user has no items in the test set, skip them in the metric calculation
        if not actual_test_items:
            continue
            
        # Get predictions based on their train data
        train_row = train_matrix.iloc[user_idx]
        recommended_items = get_top_k_recommendations(train_row, sim_matrix, k=k)
        
        # If the model couldn't recommend anything, scores are 0
        if not recommended_items:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
            
        # Count how many recommended items are in the actual test items
        hits = len(set(recommended_items).intersection(set(actual_test_items)))
        
        # Calculate Precision and Recall for this specific user
        user_precision = hits / k
        user_recall = hits / len(actual_test_items)
        
        precisions.append(user_precision)
        recalls.append(user_recall)
        
    # Average the metrics across all valid users
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    
    # Calculate F1 score
    if mean_precision + mean_recall > 0:
        mean_f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
    else:
        mean_f1 = 0.0
    
    print(f"\n--- Evaluation Results (Top {k} Recommendations) ---")
    print(f"Precision@{k}: {mean_precision:.4f} ({mean_precision*100:.2f}%)")
    print(f"Recall@{k}:    {mean_recall:.4f} ({mean_recall*100:.2f}%)")
    print(f"F1@{k}:        {mean_f1:.4f} ({mean_f1*100:.2f}%)")
    
    return mean_precision, mean_recall

# Execute the evaluation
# Testing how well it performs when giving 2 recommendations
evaluate_model(train_df, test_df, train_item_sim_df, k=2)
evaluate_model(train_df, test_df, train_item_sim_df, k=3)
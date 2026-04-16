from sklearn.neighbors import NearestNeighbors

# ========================= 7. User-Based Collaborative Filtering ==========================
print("\n--- Building User-Based Collaborative Filtering Model ---")

# 1. Isolate the User-Item Matrix
# We only want the product holding columns for the pure collaborative filtering approach
product_cols = [col for col in prime_df.columns if col.startswith('HAS_PROD_')]

# Drop duplicates just in case, and set RIMNO as the index for easy lookups
ui_matrix = prime_df.drop_duplicates(subset=['RIMNO']).set_index('RIMNO')[product_cols]

# 2. Fit the Nearest Neighbors Model
# We use 'cosine' distance to find users with similar product portfolios.
# n_neighbors is set to 11 because the closest neighbor is always the user themselves.
knn_recommender = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=11)
knn_recommender.fit(ui_matrix)
print("Collaborative Filtering model fitted successfully.")

def get_product_recommendations(target_rimno, ui_matrix, model, top_n_recs=3):
    """
    Finds similar users and recommends products the target user doesn't already have.
    """
    if target_rimno not in ui_matrix.index:
        return f"User {target_rimno} not found in the matrix."

    # Get the target user's current holdings
    user_vector = ui_matrix.loc[[target_rimno]]
    products_owned = ui_matrix.columns[user_vector.values.flatten() > 0].tolist()

    # Find the nearest neighbors (distances and indices)
    distances, indices = model.kneighbors(user_vector)
    
    # Flatten arrays and ignore the first item (which is the user themselves, distance 0)
    neighbor_distances = distances.flatten()[1:]
    neighbor_indices = indices.flatten()[1:]
    
    # Convert distances to similarity scores (Cosine Similarity = 1 - Cosine Distance)
    similarities = 1 - neighbor_distances

    # Get the holdings of these similar users
    neighbor_holdings = ui_matrix.iloc[neighbor_indices]
    
    # Weight the neighbors' holdings by their similarity score
    # A neighbor with 0.9 similarity has more vote weight than one with 0.5
    weighted_holdings = neighbor_holdings.T.dot(similarities)
    
    # Filter out the products the target user already owns
    recommendations = weighted_holdings.drop(labels=products_owned, errors='ignore')
    
    # Sort by highest recommendation score and get the Top N
    top_recommendations = recommendations.sort_values(ascending=False).head(top_n_recs)
    
    # Clean up the output formatting
    result = top_recommendations[top_recommendations > 0].to_dict()
    clean_result = {k.replace('HAS_PROD_', ''): round(v, 3) for k, v in result.items()}
    
    return clean_result

# ========================= 8. Testing the Recommender ==========================
# Let's test it on a random existing customer
sample_user = ui_matrix.index[0] 
print(f"\n--- Recommendations for RIMNO: {sample_user} ---")

# Show what they currently own
owned = [col.replace('HAS_PROD_', '') for col in ui_matrix.columns if ui_matrix.loc[sample_user, col] > 0]
print(f"Currently Owns: {owned}")

# Generate Recommendations
recs = get_product_recommendations(target_rimno=sample_user, ui_matrix=ui_matrix, model=knn_recommender, top_n_recs=3)
print(f"Recommended Products (Weighted Score): {recs}")

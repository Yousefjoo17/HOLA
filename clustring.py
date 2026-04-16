import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# 1. Isolate clustering features by dropping IDs, dates, and product holdings
cols_to_drop = ['RIMNO', 'BRANCH_ID', 'BRANCH_NAME', 'CLOSURE_DATE', 'CREATION_DATE', 
                'LAST_PAYMENT_DATE', 'LAST_STAEMENT_DATE', 'DOB', 'PRODUCT_NAME']
prod_cols = [col for col in prime_df.columns if 'HAS_PROD_' in col]

# Ensure we only pass numeric data into the model
features_df = prime_df.drop(columns=cols_to_drop + prod_cols, errors='ignore')
features_df = features_df.select_dtypes(include=[np.number])

# ========================= Missing Values Handling ==========================
print("\n--- Checking for Missing Values before PCA ---")
missing_before_pca = features_df.isna().sum()
missing_cols = missing_before_pca[missing_before_pca > 0]

if not missing_cols.empty:
    print("Warning: Missing values found. Imputing with median...")
    imputer = SimpleImputer(strategy='median')
    # Fit and transform, then convert back to DataFrame to keep column names
    features_to_use = pd.DataFrame(
        imputer.fit_transform(features_df), 
        columns=features_df.columns, 
        index=features_df.index
    )
    print("Imputation complete.")
else:
    print("No missing values found. Data is ready for PCA.")
    features_to_use = features_df
# ============================================================================

# 2. Dimensionality Reduction (Keep 95% of the variance)
# Notice we are passing 'features_to_use' here, which is guaranteed to be null-free
pca = PCA(n_components=0.95, random_state=42)
reduced_features = pca.fit_transform(features_to_use)
print(f"Reduced features from {features_to_use.shape[1]} to {reduced_features.shape[1]} dimensions.")

# 3. Use the Elbow Method to find the optimal 'k'
inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(reduced_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# 4. Fit final K-Means model
optimal_k = 4 # Adjust this based on your visual interpretation of the elbow plot
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
prime_df['CLUSTER'] = final_kmeans.fit_predict(reduced_features)

# 5. Profile the clusters (Inverse transform if you want to see unscaled raw numbers)
# Let's look at the average behaviors per cluster to name them (e.g., "Wealthy Travelers", "Low-Spend Youth")
profiling_cols = ['TOTAL_SPEND_AMT', 'AVG_TRXN_AMT', 'AGE', 'CREDIT_LIMIT', 'FOREIGN_SPEND_RATIO', 'FEE_TO_LIMIT_RATIO']

# Safeguard: Only profile columns that actually exist in the dataframe
actual_profiling_cols = [col for col in profiling_cols if col in prime_df.columns]

cluster_summary = prime_df.groupby('CLUSTER')[actual_profiling_cols].mean()
print("\n--- Cluster Profiles ---")
print(cluster_summary)


# 6. Calculate product popularity within each cluster
cluster_product_popularity = prime_df.groupby('CLUSTER')[prod_cols].mean()

def recommend_products(user_rimno, df, popularity_matrix, top_n=2):
    """
    Recommends products to a user based on what is popular in their cluster.
    """
    # Isolate the user's data
    user_data = df[df['RIMNO'] == user_rimno]
    if user_data.empty:
        return "User not found."
    
    user_data = user_data.iloc[0]
    user_cluster = user_data['CLUSTER']
    
    # Sort products by popularity in this user's cluster (highest to lowest)
    cluster_prefs = popularity_matrix.loc[user_cluster].sort_values(ascending=False)
    
    recommendations = []
    
    # Iterate through popular products
    for prod in cluster_prefs.index:
        # If the user's holding for this product is 0 (they don't have it)
        if user_data[prod] == 0:
            # Clean up the string to get the actual product name
            clean_name = prod.replace('HAS_PROD_', '')
            recommendations.append(clean_name)
            
        # Stop once we hit our desired number of recommendations
        if len(recommendations) == top_n:
            break
            
    return recommendations

# --- Example Usage ---
# Let's recommend 2 products for the first user in your dataframe
sample_user = prime_df['RIMNO'].iloc[0]
recs = recommend_products(sample_user, prime_df, cluster_product_popularity, top_n=2)
print(f"\nRecommended products for RIMNO {sample_user}: {recs}")

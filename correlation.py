from scipy.stats import chi2_contingency, f_oneway

print("\n▶ Analyzing Product (DESCRIPTION) Relationships …")

# --- 1. DEFINING FEATURE SPACES ---
target_col = "DESCRIPTION"
trxn_categorical_cols = ["CCY", "MCC", "SETTLEMENT CCY", "BANKBRANCH", "TRXN COUNTRY", "REVERSAL FLAG"]
txn_flt = ["ORIG AMOUNT", "EMBEDDED _FEE", "BILLING AMT", "SETTLEMENT AMT"]

# --- 2. CRAMER'S V FOR CATEGORICAL CORRELATION ---
def cramers_v(x, y):
    """Calculates Cramer's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty or confusion_matrix.size == 0:
        return np.nan
    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    
    # Avoid division by zero
    if min(k-1, r-1) == 0:
        return 0.0
        
    return np.sqrt(chi2 / (n * min(k-1, r-1)))

print("\n--- Categorical Association with DESCRIPTION (Cramer's V) ---")
cramers_results = {}
for col in trxn_categorical_cols:
    if col in transaction_df.columns:
        # Drop NaNs to ensure accurate matrix calculation
        valid_data = transaction_df[[target_col, col]].dropna()
        score = cramers_v(valid_data[target_col], valid_data[col])
        cramers_results[col] = score
        print(f"{col.ljust(20)} : {score:.4f} (0=None, 1=Perfect)")

# --- 3. COUNTS & CROSS-TABULATION PLOTS ---
# Let's plot the relationship for the top categorical features 
# (e.g. MCC vs DESCRIPTION)
print("\nGenerating categorical count plots...")
for col in ["MCC", "TRXN COUNTRY"]:  # Focusing on the most likely drivers for recommendations
    if col in transaction_df.columns:
        # Get top 10 most frequent categories to avoid cluttered charts
        top_cats = transaction_df[col].value_counts().nlargest(10).index
        plot_data = transaction_df[transaction_df[col].isin(top_cats)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=plot_data, x=col, hue=target_col, palette=PALETTE, ax=ax)
        plt.title(f"Product Usage Distribution across Top {col}s")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Product (DESCRIPTION)", bbox_to_anchor=(1.05, 1), loc='upper left')
        save(fig, f"product_counts_by_{col.replace(' ', '_')}.png")


# --- 4. ANOVA FOR NUMERICAL FEATURES ---
print("\n--- Numerical Impact on DESCRIPTION (ANOVA) ---")
for col in txn_flt:
    if col in transaction_df.columns:
        # Group the numeric feature by the target categories
        groups = [group[col].dropna().values for name, group in transaction_df.groupby(target_col) if len(group[col].dropna()) > 0]
        
        if len(groups) > 1:
            f_stat, p_val = f_oneway(*groups)
            # If p-val < 0.05, the numerical amount heavily influences the product used
            print(f"{col.ljust(20)} : F-stat={f_stat:.2f}, p-value={p_val:.4e}")
            
            # Generate a boxplot to visualize the numerical differences between products
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=transaction_df, x=target_col, y=col, palette=PALETTE, ax=ax, showfliers=False)
            plt.title(f"Distribution of {col} by Product")
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(col)
            save(fig, f"product_numdist_{col.replace(' ', '_')}.png")
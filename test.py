import numpy as np

print("Compressing data types to save RAM...")
# Convert features to 32-bit floats (cuts memory in half)
X = X.astype(np.float32)

# Convert one-hot encoded targets to 8-bit integers (cuts memory by 8x!)
y = y.astype(np.int8)

print("Compression complete!")


print("Training constrained Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=50,      # Start with 50 trees instead of 100
    max_depth=20,         # Stop trees from growing infinitely deep
    n_jobs=2,             # Only use 2 cores (prevents massive data copying)
    random_state=42
)
rf_model.fit(X_train, y_train)
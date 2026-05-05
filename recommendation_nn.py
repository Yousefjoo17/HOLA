import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')

print("============================================================")
print(" PHASE 6: Advanced Multi-Label Deep Learning (PyTorch)")
print("============================================================\n")

# ==========================================
# 1. Load and Prepare Data
# ==========================================
print("Loading final_customer_profile_optimized.csv...")
try:
    df = pd.read_csv("final_customer_profile_optimized.csv")
except FileNotFoundError:
    print("Error: 'final_customer_profile_optimized.csv' not found.")
    exit()

target_cols = [col for col in df.columns if col.startswith('HAS_PROD_')]
exclude_cols = ['CUSTOMER_ID', 'BRANCH_ID'] + target_cols
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Fill NaNs and convert to numpy arrays
X = df[feature_cols].fillna(0).values
Y = df[target_cols].values

print(f"Features: {X.shape[1]} | Targets: {Y.shape[1]}")

# Train/Test Split (Global split)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Neural Networks require normalized features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 2. Calculate Class Imbalance Weights
# ==========================================
# For each product: weight = (Total Negatives) / (Total Positives)
pos_weights = []
for i in range(Y_train.shape[1]):
    positives = Y_train[:, i].sum()
    negatives = len(Y_train) - positives
    # Add a small epsilon to avoid division by zero for extremely rare products
    weight = negatives / (positives + 1e-5) 
    pos_weights.append(weight)

# Convert to tensor for the loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
print(f"\nUsing Device: {device}")

# ==========================================
# 3. Create PyTorch Datasets & DataLoaders
# ==========================================
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))

# Batch size controls how many customers the network looks at before updating weights
BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 4. Define the Shared-Bottom Neural Network
# ==========================================
class Customer360Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Customer360Net, self).__init__()
        
        # Shared Layers: Learns the global customer profile and product correlations
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3), # Prevents overfitting
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Output Layer: 57 neurons, one for each product
        self.output = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.shared(x)
        logits = self.output(x)
        # We DO NOT apply Sigmoid here because BCEWithLogitsLoss applies it automatically (more numerically stable)
        return logits

model = Customer360Net(input_dim=X.shape[1], num_classes=Y.shape[1]).to(device)

# Loss function customized with our specific positive weights to fix Macro F1
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 5. Training Loop
# ==========================================
EPOCHS = 20

print("\nStarting Training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(batch_X)
        
        # Calculate loss
        loss = criterion(logits, batch_Y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Training Loss: {avg_loss:.4f}")

# ==========================================
# 6. Evaluation
# ==========================================
print("\n============================================================")
print(" FINAL EVALUATION")
print("============================================================\n")

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        batch_X = batch_X.to(device)
        
        logits = model(batch_X)
        # Apply sigmoid to convert raw logits to probabilities (0.0 to 1.0)
        probs = torch.sigmoid(logits)
        
        # Convert probabilities to binary predictions (threshold = 0.5)
        # You can tune this threshold later to favor Precision or Recall
        preds = (probs > 0.5).float().cpu().numpy()
        
        all_preds.append(preds)
        all_targets.append(batch_Y.numpy())

# Stack all batches into final matrices
Y_pred_final = np.vstack(all_preds)
Y_test_final = np.vstack(all_targets)

# Calculate Global Metrics
micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(Y_test_final, Y_pred_final, average='micro', zero_division=0)
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(Y_test_final, Y_pred_final, average='macro', zero_division=0)

print("--- MICRO AVERAGE ---")
print(f"Precision: {micro_p:.4f}")
print(f"Recall:    {micro_r:.4f}")
print(f"F1-Score:  {micro_f1:.4f}\n")

print("--- MACRO AVERAGE ---")
print(f"Precision: {macro_p:.4f}")
print(f"Recall:    {macro_r:.4f}")
print(f"F1-Score:  {macro_f1:.4f}\n")
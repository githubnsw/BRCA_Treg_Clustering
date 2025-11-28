"""
Autoencoder for Dimensionality Reduction of Multi-omics Data
=============================================================
Description: Train autoencoder to extract latent features from high-dimensional omics data
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# -------------------- Configuration --------------------
INPUT_PATH = "data/"
OUTPUT_PATH = "results/ae/"

BATCH_SIZE = 32
NUM_EPOCHS = 200
PATIENCE = 5
LEARNING_RATE = 0.001
TEST_SIZE = 0.1
SEED = 42

ENCODER_DIMS = [4000, 1500, 800, 400]  # Hidden layer dimensions
# -------------------------------------------------------


def set_seed(seed=42):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=5, delta=0, save_path="best_model.pt"):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_best_model(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_best_model(self, model):
        torch.save(model.state_dict(), self.save_path)
        print(f"Best model saved with validation loss: {self.best_loss:.4f}")


class Autoencoder(nn.Module):
    """Symmetric autoencoder for dimensionality reduction."""
    
    def __init__(self, input_dim, hidden_dims=None):
        super(Autoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = ENCODER_DIMS
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims[:-1]:
            encoder_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (symmetric)
        decoder_layers = []
        decoder_dims = hidden_dims[::-1]
        prev_dim = decoder_dims[0]
        for dim in decoder_dims[1:]:
            decoder_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        decoder_layers.extend([nn.Linear(prev_dim, input_dim)])
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder(model, train_loader, test_loader, device, save_path):
    """Train autoencoder with early stopping."""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(patience=PATIENCE, save_path=save_path)
    
    train_losses, val_losses = [], []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, decoded = model(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                _, decoded = model(batch)
                val_loss += criterion(decoded, batch).item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return train_losses, val_losses


def plot_loss(train_losses, val_losses, save_path):
    """Plot and save training/validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_list = glob.glob(os.path.join(INPUT_PATH, "*.csv"))
    print(f"Found {len(file_list)} files to process")
    
    for file_path in file_list:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\nProcessing: {file_name}")
        
        # Load data
        data = pd.read_csv(file_path, index_col=0)
        data_transposed = data.T
        
        # Split data
        X_train, X_test = train_test_split(
            data_transposed, test_size=TEST_SIZE, random_state=SEED
        )
        
        X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
        
        train_loader = torch.utils.data.DataLoader(
            X_train_tensor, batch_size=BATCH_SIZE, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            X_test_tensor, batch_size=BATCH_SIZE, shuffle=False
        )
        
        # Initialize model
        set_seed(SEED)
        model = Autoencoder(X_train_tensor.shape[1]).to(device)
        
        # Train
        model_save_path = os.path.join(OUTPUT_PATH, f"best_model_{file_name}.pt")
        train_losses, val_losses = train_autoencoder(
            model, train_loader, test_loader, device, model_save_path
        )
        
        # Load best model and extract features
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        
        with torch.no_grad():
            data_tensor = torch.tensor(
                data_transposed.to_numpy(), dtype=torch.float32
            ).to(device)
            latent_features = model.encoder(data_tensor)
        
        # Save reduced data
        latent_df = pd.DataFrame(
            latent_features.cpu().numpy(), 
            index=data_transposed.index
        )
        output_file = os.path.join(OUTPUT_PATH, f"AE_reduced_{file_name}.csv")
        latent_df.to_csv(output_file, index=True)
        print(f"Reduced data saved: {output_file}")
        
        # Save loss plot
        plot_path = os.path.join(OUTPUT_PATH, f"AE_loss_{file_name}.png")
        plot_loss(train_losses, val_losses, plot_path)
        print(f"Loss plot saved: {plot_path}")


if __name__ == "__main__":
    main()

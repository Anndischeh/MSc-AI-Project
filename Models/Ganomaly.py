import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bars
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# --- Configuration ---
DATA_PATH = '/kaggle/working/BraTS2021_Training_Data'  # Change this to your correct path
RESULTS_DIR = './results'
N_PATIENTS_TO_USE = 100  # Use a subset for faster testing. Set to a large number (e.g., 2000) to use all.
MODALITY = 't1ce'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 1  # set to 100 later
Z_DIM = 128  # Latent dimension
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)

# NOTE: BraTSDataset is assumed to be already defined/imported
# from your_ae_file import BraTSDataset

# ==============================================================================
# 1. Encoder, Generator, and Discriminator Networks for GANomaly
# ==============================================================================

# Encoder - Extracts the latent representation of the image
class Encoder(nn.Module):
    def __init__(self, img_size, z_dim):
        super(Encoder, self).__init__()
        self.img_size = img_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * (img_size // 8) * (img_size // 8), z_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        z = self.fc1(x)
        return z

# Generator - Takes latent vector z and generates an image
class Generator(nn.Module):
    def __init__(self, img_size, z_dim):
        super(Generator, self).__init__()
        self.img_size = img_size
        feature_side = img_size // 8  # 128 -> 16
        self.fc1 = nn.Linear(z_dim, 128 * feature_side * feature_side)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        feature_side = self.img_size // 8
        x = torch.relu(self.fc1(z))
        x = x.view(x.size(0), 128, feature_side, feature_side)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))  # Use tanh as the output activation function (images in [-1,1])
        return x

# Discriminator - Classifies images as real or fake
class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * (img_size // 8) * (img_size // 8), 1)
        self.leaky_relu = nn.LeakyReLU(0.2)  # Use LeakyReLU as a layer instead of torch.leaky_relu

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the output to pass through the fully connected layer
        x = torch.sigmoid(self.fc1(x))  # Output a probability (real or fake)
        return x

# ==============================================================================
# 2. GANomaly Loss Function
# ==============================================================================

def ganomaly_loss(reconstructed_x, x, real_images, discriminator, encoder, generator, lambda_kl=0.1):
    # Reconstruction loss (autoencoder part)
    recon_loss = nn.MSELoss()(reconstructed_x, x)

    # Latent/consistency term (we call it kl_loss here)
    z = encoder(x)
    generated_images = generator(z)
    kl_loss = nn.MSELoss()(generated_images, real_images)

    # Discriminator loss
    real_labels = torch.ones(real_images.size(0), 1).to(DEVICE)
    fake_labels = torch.zeros(real_images.size(0), 1).to(DEVICE)
    
    real_output = discriminator(real_images)
    fake_output = discriminator(generated_images.detach())  # Discriminator doesn't backprop through generator

    real_loss = nn.BCELoss()(real_output, real_labels)
    fake_loss = nn.BCELoss()(fake_output, fake_labels)

    # Total discriminator loss
    d_loss = real_loss + fake_loss

    # Generator loss (fool the discriminator)
    g_loss = nn.BCELoss()(fake_output, real_labels)

    total_loss = recon_loss + lambda_kl * kl_loss + d_loss + g_loss
    return total_loss, recon_loss, kl_loss, d_loss, g_loss

# ==============================================================================
# 3. Training and Evaluation Functions for GANomaly
# ==============================================================================

def train_ganomaly(generator, discriminator, encoder, train_loader,
                   optimizer_g, optimizer_d, optimizer_e, lambda_kl, device, epochs):
    generator.train()
    discriminator.train()
    encoder.train()

    for epoch in range(epochs):
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        running_d_loss = 0.0
        running_g_loss = 0.0
        for data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            data = data.to(device)

            # --- Train Encoder, Generator, and Discriminator ---
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            optimizer_e.zero_grad()

            z = encoder(data)
            generated_images = generator(z)

            total_loss, recon_loss, kl_loss, d_loss, g_loss = ganomaly_loss(
                generated_images, data, data, discriminator, encoder, generator, lambda_kl
            )

            total_loss.backward()
            optimizer_g.step()
            optimizer_d.step()
            optimizer_e.step()

            running_loss += total_loss.item()
            running_recon_loss += recon_loss.item()
            running_kl_loss += kl_loss.item()
            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Loss: {running_loss / len(train_loader):.4f}, "
            f"Recon Loss: {running_recon_loss / len(train_loader):.4f}, "
            f"KL Loss: {running_kl_loss / len(train_loader):.4f}, "
            f"D Loss: {running_d_loss / len(train_loader):.4f}, "
            f"G Loss: {running_g_loss / len(train_loader):.4f}"
        )

def evaluate_ganomaly(encoder, generator, test_loader, device):
    """
    Anomaly score: reconstruction error ||x - G(E(x))||_1
    Higher score => more anomalous
    """
    encoder.eval()
    generator.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for data, label in tqdm(test_loader, desc="Evaluating GANomaly", leave=False):
            data = data.to(device)
            z = encoder(data)
            reconstructed = generator(z)
            error = torch.mean(torch.abs(data - reconstructed), dim=(1, 2, 3))
            scores.extend(error.cpu().numpy())
            labels.extend(label.cpu().numpy())

    return np.array(scores), np.array(labels)

def plot_reconstruction_error(errors, labels, results_dir, model_name="ganomaly"):
    plt.figure(figsize=(10, 6))
    plt.hist(errors[labels == 0], bins=50, alpha=0.7, label='Normal Slices', density=True)
    plt.hist(errors[labels == 1], bins=50, alpha=0.7, label='Anomalous Slices', density=True)
    
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.title(f'Reconstruction Error Distribution - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_reconstruction_error_distribution.png'))
    plt.close()

def compute_and_save_metrics(scores, labels, model_name, results_dir, num_sample_outputs=100):
    """
    scores: anomaly scores (higher = more anomalous)
    labels: ground truth (0 = normal, 1 = anomaly)
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if len(np.unique(labels)) < 2:
        print("⚠️ Warning: Only one class in labels, cannot compute ROC-AUC.")
        roc_auc = np.nan
        fpr, tpr, thresholds = np.array([]), np.array([]), np.array([])
    else:
        roc_auc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)

    # Save raw arrays
    np.save(os.path.join(results_dir, f"{model_name}_scores.npy"), scores)
    np.save(os.path.join(results_dir, f"{model_name}_labels.npy"), labels)
    np.save(os.path.join(results_dir, f"{model_name}_fpr.npy"), fpr)
    np.save(os.path.join(results_dir, f"{model_name}_tpr.npy"), tpr)
    np.save(os.path.join(results_dir, f"{model_name}_thresholds.npy"), thresholds)

    # Plot ROC curve (if computable)
    if len(fpr) > 0:
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{model_name}_roc_curve.png"))
        plt.close()

    # Sample outputs (for thesis tables)
    n = min(num_sample_outputs, len(scores))
    idx = np.arange(len(scores))[:n]
    df_samples = pd.DataFrame({
        "index": idx,
        "label": labels[idx],
        "score": scores[idx]
    })
    df_samples.to_csv(os.path.join(results_dir, f"{model_name}_sample_outputs.csv"), index=False)

    print(f"{model_name} ROC-AUC: {roc_auc:.4f}")
    print(f"Saved ROC data and {n} sample outputs to {results_dir}")

    return roc_auc

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
    print(f"!!! WARNING: No patient folders found in '{DATA_PATH}'. Please extract or provide the correct BraTS dataset.")
    
print(f"Using device: {DEVICE}")
os.makedirs(RESULTS_DIR, exist_ok=True)
    
# Load dataset
# IMPORTANT: train only on normal slices for unsupervised anomaly detection
train_dataset = BraTSDataset(
    root_dir=DATA_PATH,
    num_patients=N_PATIENTS_TO_USE,
    img_size=IMG_SIZE,
    train=True,
    normal_only=True
)
test_dataset = BraTSDataset(
    root_dir=DATA_PATH,
    num_patients=N_PATIENTS_TO_USE,
    img_size=IMG_SIZE,
    train=False,
    normal_only=False
)

if len(train_dataset) == 0:
    print("\nFATAL: Training dataset is empty. Check DATA_PATH and N_PATIENTS_TO_USE.")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize models, optimizers
encoder = Encoder(img_size=IMG_SIZE, z_dim=Z_DIM).to(DEVICE)
generator = Generator(img_size=IMG_SIZE, z_dim=Z_DIM).to(DEVICE)
discriminator = Discriminator(img_size=IMG_SIZE).to(DEVICE)

optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
optimizer_e = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

lambda_kl = 0.1  # Weight for the KL/consistency term

# Train GANomaly model
train_ganomaly(generator, discriminator, encoder,
               train_loader, optimizer_g, optimizer_d, optimizer_e,
               lambda_kl, DEVICE, EPOCHS)

# Evaluate GANomaly
errors_ganomaly, labels_ganomaly = evaluate_ganomaly(encoder, generator, test_loader, DEVICE)

# Plot reconstruction error distribution
model_name_ganomaly = "GANomaly"
plot_reconstruction_error(errors_ganomaly, labels_ganomaly, RESULTS_DIR, model_name=model_name_ganomaly.lower())

# Compute ROC/AUC & save metrics + sample outputs
roc_auc_ganomaly = compute_and_save_metrics(
    scores=errors_ganomaly,
    labels=labels_ganomaly,
    model_name=model_name_ganomaly,
    results_dir=RESULTS_DIR,
    num_sample_outputs=100
)

# Save model weights
torch.save(encoder.state_dict(), os.path.join(RESULTS_DIR, f"{model_name_ganomaly}_encoder.pth"))
torch.save(generator.state_dict(), os.path.join(RESULTS_DIR, f"{model_name_ganomaly}_generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(RESULTS_DIR, f"{model_name_ganomaly}_discriminator.pth"))

# Simple comparison CSV for this script
df_results = pd.DataFrame([{"model": model_name_ganomaly, "roc_auc": float(roc_auc_ganomaly)}])
df_results.to_csv(os.path.join(RESULTS_DIR, "model_comparison_roc_auc_ganomaly_only.csv"), index=False)

print("\n=== Model Comparison (ROC-AUC, GANomaly only) ===")
print(df_results)
print(f"\nSaved GANomaly ROC-AUC table to: {os.path.join(RESULTS_DIR, 'model_comparison_roc_auc_ganomaly_only.csv')}")

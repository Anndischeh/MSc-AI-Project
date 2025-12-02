import os
import glob
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
DATA_PATH = '/kaggle/working/BraTS2021_Training_Data'  # Change this to your correct path
RESULTS_DIR = './results'
N_PATIENTS_TO_USE = 100  # Use a subset for faster testing. Set to a large number (e.g., 2000) to use all.
MODALITY = 't1ce'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 1  # set to 100 later
Z_DIM = 128
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. DATASET (same idea as AE script)
# ==============================================================================

class BraTSDataset(Dataset):
    def __init__(self, root_dir, num_patients, img_size=128, train=True, normal_only=False):
        self.root_dir = root_dir
        self.img_size = img_size
        self.train = train
        self.normal_only = normal_only

        # Find all patient directories
        all_patient_dirs = sorted(glob.glob(os.path.join(root_dir, "BraTS*")))
        if not all_patient_dirs:
            print(f"!!! WARNING: No patient folders found in '{root_dir}'. Please ensure the path is correct.")
            print("Please extract or provide the correct BraTS dataset.")
            raise FileNotFoundError(f"No patient folders found in {root_dir}.")
        
        # Use a subset of patients as specified
        all_patient_dirs = all_patient_dirs[:num_patients]

        # Train/validation split on PATIENTS to prevent data leakage
        train_dirs, val_dirs = train_test_split(all_patient_dirs, test_size=0.2, random_state=42)
        
        if self.train:
            self.patient_dirs = train_dirs
        else:
            self.patient_dirs = val_dirs

        print(f"Found {len(self.patient_dirs)} patients for this set.")
        self.slices = self._create_slice_list()
        
        if self.normal_only:
            self.slices = [s for s in self.slices if s['label'] == 0]
            print(f"Filtered to {len(self.slices)} normal slices for training.")

    def _create_slice_list(self):
        slice_list = []
        if not self.patient_dirs:
            return slice_list
            
        print("Creating slice manifest...")
        for patient_dir in tqdm(self.patient_dirs, desc="Patient Folders"):
            try:
                mod_path = glob.glob(os.path.join(patient_dir, f'*_{MODALITY}.nii.gz'))[0]
                seg_path = glob.glob(os.path.join(patient_dir, '*_seg.nii.gz'))[0]
            except IndexError:
                continue  # If files are missing, skip this patient
            
            mod_vol = nib.load(mod_path).get_fdata()
            seg_vol = nib.load(seg_path).get_fdata()

            for slice_idx in range(mod_vol.shape[2]):
                mod_slice = mod_vol[:, :, slice_idx]
                
                # Filter out uninformative slices
                if np.mean(mod_slice) < 10 or np.std(mod_slice) < 10:
                    continue
                
                seg_slice = seg_vol[:, :, slice_idx]
                is_abnormal = np.any(seg_slice > 0)
                
                slice_info = {
                    'patient_dir': patient_dir,
                    'slice_idx': slice_idx,
                    'label': 1 if is_abnormal else 0
                }
                slice_list.append(slice_info)
                
        print(f"Found {len(slice_list)} valid slices after filtering.")
        return slice_list

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice_info = self.slices[idx]
        patient_dir = slice_info['patient_dir']
        slice_idx = slice_info['slice_idx']
        label = slice_info['label']

        mod_path = glob.glob(os.path.join(patient_dir, f'*_{MODALITY}.nii.gz'))[0]
        mod_vol = nib.load(mod_path).get_fdata()
        
        slice_2d = mod_vol[:, :, slice_idx]
        slice_resized = cv2.resize(slice_2d, (self.img_size, self.img_size))
        
        # Normalize to [-1, 1]
        slice_min = slice_resized.min()
        slice_max = slice_resized.max()
        if slice_max > slice_min:
            slice_normalized = (slice_resized - slice_min) / (slice_max - slice_min)
        else:
            slice_normalized = slice_resized * 0.0
        
        slice_scaled = (slice_normalized * 2.0) - 1.0
        img_final = slice_scaled[np.newaxis, ...].astype(np.float32)

        return torch.from_numpy(img_final), torch.tensor(label, dtype=torch.long)

# ==============================================================================
# 2. VAE ARCHITECTURE
# ==============================================================================

class VAE_Encoder(nn.Module):
    def __init__(self, img_size, z_dim):
        super(VAE_Encoder, self).__init__()
        self.img_size = img_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        self.fc1 = nn.Linear(128 * (img_size // 8) * (img_size // 8), z_dim)  # mean
        self.fc2 = nn.Linear(128 * (img_size // 8) * (img_size // 8), z_dim)  # log variance

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        mean = self.fc1(x)
        log_var = self.fc2(x)
        return mean, log_var

class VAE_Decoder(nn.Module):
    def __init__(self, img_size, z_dim):
        super(VAE_Decoder, self).__init__()
        self.img_size = img_size
        self.fc1 = nn.Linear(z_dim, 128 * (img_size // 8) * (img_size // 8))
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = x.view(x.size(0), 128, self.img_size // 8, self.img_size // 8)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

class VAE(nn.Module):
    def __init__(self, img_size, z_dim):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(img_size, z_dim)
        self.decoder = VAE_Decoder(img_size, z_dim)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mean, log_var

# ==============================================================================
# 3. LOSS & HELPERS
# ==============================================================================

def vae_loss(reconstructed_x, x, mean, log_var):
    recon_loss = nn.MSELoss()(reconstructed_x, x)
    kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

def train_vae(model, train_loader, optimizer, device, epochs, model_name="VAE"):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        for data, _ in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}", leave=False):
            data = data.to(device)
            
            optimizer.zero_grad()
            reconstructed_data, mean, log_var = model(data)
            loss, recon_loss, kl_loss = vae_loss(reconstructed_data, data, mean, log_var)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            running_kl_loss += kl_loss.item()

        print(
            f"{model_name} - Epoch [{epoch+1}/{epochs}], "
            f"Loss: {running_loss / len(train_loader):.4f}, "
            f"Recon: {running_recon_loss / len(train_loader):.4f}, "
            f"KL: {running_kl_loss / len(train_loader):.4f}"
        )

def evaluate_vae(model, test_loader, device):
    model.eval()
    reconstruction_errors = []
    labels = []
    with torch.no_grad():
        for data, label in tqdm(test_loader, desc="Evaluating VAE", leave=False):
            data = data.to(device)
            reconstructed_data, _, _ = model(data)
            error = torch.mean(torch.abs(data - reconstructed_data), dim=(1, 2, 3))
            reconstruction_errors.extend(error.cpu().numpy())
            labels.extend(label.cpu().numpy())

    reconstruction_errors = np.array(reconstruction_errors)
    labels = np.array(labels)
    return reconstruction_errors, labels

def plot_reconstruction_error(errors, labels, results_dir, model_name="vae"):
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
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    roc_auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Save raw arrays
    np.save(os.path.join(results_dir, f"{model_name}_scores.npy"), scores)
    np.save(os.path.join(results_dir, f"{model_name}_labels.npy"), labels)
    np.save(os.path.join(results_dir, f"{model_name}_fpr.npy"), fpr)
    np.save(os.path.join(results_dir, f"{model_name}_tpr.npy"), tpr)
    np.save(os.path.join(results_dir, f"{model_name}_thresholds.npy"), thresholds)

    # Plot ROC curve
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

    # Sample outputs
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
# 4. MAIN EXECUTION
# ==============================================================================

if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
    print(f"!!! WARNING: No patient folders found in '{DATA_PATH}'. Please extract or provide the correct BraTS dataset.")

print(f"Using device: {DEVICE}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Datasets & loaders
# IMPORTANT: train only on normal slices for anomaly detection
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
test_loader_vae = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model name for saving/comparison
model_name_vae = "VAE"

# Initialize model & optimizer
model_vae = VAE(img_size=IMG_SIZE, z_dim=Z_DIM).to(DEVICE)
optimizer_vae = optim.Adam(model_vae.parameters(), lr=LEARNING_RATE)

# Train
train_vae(model_vae, train_loader, optimizer_vae, DEVICE, EPOCHS, model_name=model_name_vae)

# Evaluate -> reconstruction error as anomaly score
errors_vae, labels_vae = evaluate_vae(model_vae, test_loader_vae, DEVICE)

# Plot reconstruction error distribution
plot_reconstruction_error(errors_vae, labels_vae, RESULTS_DIR, model_name=model_name_vae.lower())

# Compute ROC/AUC & save metrics + sample outputs
roc_auc_vae = compute_and_save_metrics(
    scores=errors_vae,
    labels=labels_vae,
    model_name=model_name_vae,
    results_dir=RESULTS_DIR,
    num_sample_outputs=100
)

# Save model weights
torch.save(model_vae.state_dict(), os.path.join(RESULTS_DIR, f"{model_name_vae}_model.pth"))

# Simple comparison table for THIS script (you can merge across models later)
df_results = pd.DataFrame([{"model": model_name_vae, "roc_auc": float(roc_auc_vae)}])
df_results.to_csv(os.path.join(RESULTS_DIR, "model_comparison_roc_auc_vae_only.csv"), index=False)

print("\n=== Model Comparison (ROC-AUC, this script) ===")
print(df_results)
print(f"\nSaved VAE comparison table to: {os.path.join(RESULTS_DIR, 'model_comparison_roc_auc_vae_only.csv')}")

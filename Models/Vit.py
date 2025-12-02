# VIT_with_boundary.py
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import glob
from skimage.segmentation import find_boundaries

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, roc_curve

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = '/kaggle/working/BraTS2021_Training_Data'  # same as others
RESULTS_DIR = './results'
N_PATIENTS_TO_USE = 100
MODALITY = 't1ce'     # used by BraTSDataset
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 1            # increase for real runs
Z_DIM = 256           # embedding dim (latent / token dim)
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = 16       # 128 / 16 = 8x8 patches
NUM_HEADS = 8
DEPTH = 4
DROP_RATE = 0.1
TOKEN_DROP_RATE = 0.2  # <<< NOVELTY: probability of dropping each patch token during training

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# ============================================================
# BRAIN IMAGE LOADING FUNCTIONS FOR VISUALIZATION
# ============================================================

def load_sample_brain_images_for_visualization(num_samples=5):
    """Load sample brain images with ground truth for visualization"""
    samples = []
    
    # Find patient directories
    patient_dirs = sorted(glob.glob(os.path.join(DATA_PATH, "BraTS*")))[:10]
    
    for patient_dir in patient_dirs[:num_samples]:
        try:
            mod_path = glob.glob(os.path.join(patient_dir, f'*_{MODALITY}.nii.gz'))[0]
            seg_path = glob.glob(os.path.join(patient_dir, '*_seg.nii.gz'))[0]
            
            img_volume = nib.load(mod_path).get_fdata()
            seg_volume = nib.load(seg_path).get_fdata()
            
            # Find a slice with tumor
            for slice_idx in range(img_volume.shape[2]):
                img_slice = img_volume[:, :, slice_idx]
                seg_slice = seg_volume[:, :, slice_idx]
                
                if np.any(seg_slice > 0) and np.mean(img_slice) > 20:
                    # Preprocess like in training
                    img_resized = cv2.resize(img_slice, (IMG_SIZE, IMG_SIZE))
                    
                    # Normalize to [-1, 1]
                    slice_min = img_resized.min()
                    slice_max = img_resized.max()
                    if slice_max > slice_min:
                        img_normalized = (img_resized - slice_min) / (slice_max - slice_min)
                    else:
                        img_normalized = img_resized * 0.0
                    
                    img_processed = (img_normalized * 2.0) - 1.0
                    img_tensor = torch.from_numpy(img_processed[np.newaxis, np.newaxis, ...]).float()
                    
                    # Process segmentation
                    seg_resized = cv2.resize(seg_slice, (IMG_SIZE, IMG_SIZE))
                    seg_binary = (seg_resized > 0).astype(np.float32)
                    
                    samples.append({
                        'image': img_tensor,
                        'original': img_resized,
                        'segmentation': seg_binary,
                        'patient': os.path.basename(patient_dir),
                        'slice': slice_idx
                    })
                    break
        
        except Exception as e:
            continue
    
    return samples

def visualize_boundary_detection(model, samples, results_dir, model_name="ViT_AE_CD"):
    """Visualize tumor boundary detection on sample brain images"""
    
    model.eval()
    
    print(f"\nGenerating boundary detection visualizations for {len(samples)} samples...")
    
    for sample_idx, sample in enumerate(samples):
        print(f"  Processing sample {sample_idx + 1}...")
        
        with torch.no_grad():
            # Get reconstruction
            img_tensor = sample['image'].to(DEVICE)
            reconstruction = model(img_tensor)
            
            # Calculate reconstruction error (anomaly map)
            error = torch.abs(img_tensor - reconstruction)
            anomaly_map = error[0, 0].cpu().numpy()
            
            # Normalize anomaly map
            if anomaly_map.max() > anomaly_map.min():
                anomaly_map_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
            else:
                anomaly_map_normalized = anomaly_map
            
            # Apply threshold to get binary tumor mask
            threshold = np.percentile(anomaly_map_normalized, 95)  # Top 5% as anomalies
            tumor_mask = (anomaly_map_normalized > threshold).astype(np.float32)
            
            # Find boundaries
            gt_boundary = find_boundaries(sample['segmentation'], mode='inner')
            pred_boundary = find_boundaries(tumor_mask, mode='inner')
            
            # Create visualization figure
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # 1. Original Brain Image
            ax = axes[0, 0]
            ax.imshow(sample['original'], cmap='gray')
            ax.set_title(f"Original Brain MRI\nPatient: {sample['patient']}\nSlice: {sample['slice']}", fontsize=10)
            ax.axis('off')
            
            # 2. Ground Truth Tumor Mask
            ax = axes[0, 1]
            ax.imshow(sample['original'], cmap='gray')
            ax.imshow(sample['segmentation'], cmap='Reds', alpha=0.3)
            ax.set_title("Ground Truth Tumor Region\n(Red overlay)", fontsize=10)
            ax.axis('off')
            
            # 3. Ground Truth Tumor Boundary
            ax = axes[0, 2]
            ax.imshow(sample['original'], cmap='gray')
            ax.contour(gt_boundary, colors='lime', linewidths=2)
            ax.set_title("Ground Truth Tumor Boundary\n(Green contour)", fontsize=10)
            ax.axis('off')
            
            # 4. Reconstruction from ViT
            ax = axes[0, 3]
            reconstruction_img = reconstruction[0, 0].cpu().numpy()
            # Denormalize from [-1, 1] to [0, 255]
            reconstruction_img = (reconstruction_img + 1) / 2 * 255
            ax.imshow(reconstruction_img, cmap='gray')
            ax.set_title("ViT Reconstruction", fontsize=10)
            ax.axis('off')
            
            # 5. Anomaly Heatmap
            ax = axes[1, 0]
            im = ax.imshow(anomaly_map_normalized, cmap='hot', vmin=0, vmax=1)
            ax.set_title("Anomaly Heatmap\n(Brighter = More Anomalous)", fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # 6. Predicted Tumor Mask
            ax = axes[1, 1]
            ax.imshow(sample['original'], cmap='gray')
            ax.imshow(tumor_mask, cmap='Reds', alpha=0.3)
            ax.set_title("Predicted Tumor Region\n(Red overlay)", fontsize=10)
            ax.axis('off')
            
            # 7. Predicted Tumor Boundary
            ax = axes[1, 2]
            ax.imshow(sample['original'], cmap='gray')
            ax.contour(pred_boundary, colors='red', linewidths=2)
            # Also show ground truth for comparison
            ax.contour(gt_boundary, colors='lime', linewidths=1, linestyles='--', alpha=0.7)
            ax.set_title("Predicted Tumor Boundary\n(Red) vs Ground Truth (Green dashed)", fontsize=10)
            ax.axis('off')
            
            # 8. Error Analysis
            ax = axes[1, 3]
            
            # Create error visualization
            error_vis = np.zeros((IMG_SIZE, IMG_SIZE, 3))
            
            # True Positives: Correctly detected tumor (green)
            true_positives = pred_boundary & gt_boundary
            error_vis[true_positives, 1] = 1.0  # Green channel
            
            # False Positives: Predicted but not in ground truth (red)
            false_positives = pred_boundary & ~gt_boundary
            error_vis[false_positives, 0] = 1.0  # Red channel
            
            # False Negatives: Missed tumor (blue)
            false_negatives = ~pred_boundary & gt_boundary
            error_vis[false_negatives, 2] = 1.0  # Blue channel
            
            ax.imshow(error_vis)
            ax.set_title("Boundary Error Analysis\nGreen: Correct, Red: FP, Blue: FN", fontsize=10)
            ax.axis('off')
            
            # Calculate boundary metrics
            if np.any(pred_boundary) and np.any(gt_boundary):
                tp = np.sum(true_positives)
                fp = np.sum(false_positives)
                fn = np.sum(false_negatives)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Add metrics to title
                plt.suptitle(f"ViT Tumor Boundary Detection - Sample {sample_idx + 1}\n"
                           f"Boundary Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}", 
                           fontsize=14, fontweight='bold', y=1.02)
            else:
                plt.suptitle(f"ViT Tumor Boundary Detection - Sample {sample_idx + 1}", 
                           fontsize=14, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            
            # Save the visualization
            output_path = os.path.join(results_dir, f"{model_name}_boundary_sample_{sample_idx + 1}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"✓ Saved {len(samples)} boundary detection visualizations")

def create_comparison_grid(samples, model, results_dir, model_name="ViT_AE_CD"):
    """Create a grid comparison of all samples"""
    
    model.eval()
    
    print(f"\nCreating comparison grid for {len(samples)} samples...")
    
    n_samples = len(samples)
    n_cols = 4  # Original, GT Boundary, Predicted Boundary, Error
    n_rows = n_samples
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    
    if n_samples == 1:
        axes = np.array([axes])
    
    for sample_idx, sample in enumerate(samples):
        with torch.no_grad():
            img_tensor = sample['image'].to(DEVICE)
            reconstruction = model(img_tensor)
            
            # Calculate anomaly map
            error = torch.abs(img_tensor - reconstruction)
            anomaly_map = error[0, 0].cpu().numpy()
            
            # Normalize
            if anomaly_map.max() > anomaly_map.min():
                anomaly_map_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
            else:
                anomaly_map_normalized = anomaly_map
            
            # Get tumor mask
            threshold = np.percentile(anomaly_map_normalized, 95)
            tumor_mask = (anomaly_map_normalized > threshold).astype(np.float32)
            
            # Find boundaries
            gt_boundary = find_boundaries(sample['segmentation'], mode='inner')
            pred_boundary = find_boundaries(tumor_mask, mode='inner')
            
            # Column 1: Original with GT boundary
            ax = axes[sample_idx, 0]
            ax.imshow(sample['original'], cmap='gray')
            ax.contour(gt_boundary, colors='lime', linewidths=2)
            ax.set_title(f"Sample {sample_idx + 1}: Patient {sample['patient'][-6:]}, Slice {sample['slice']}\nGround Truth Boundary", 
                        fontsize=10)
            ax.axis('off')
            
            # Column 2: Anomaly heatmap
            ax = axes[sample_idx, 1]
            im = ax.imshow(anomaly_map_normalized, cmap='hot')
            ax.set_title(f"Anomaly Heatmap\nThreshold: {threshold:.2f}", fontsize=10)
            ax.axis('off')
            
            # Column 3: Predicted boundary
            ax = axes[sample_idx, 2]
            ax.imshow(sample['original'], cmap='gray')
            ax.contour(pred_boundary, colors='red', linewidths=2)
            # Show ground truth for comparison
            ax.contour(gt_boundary, colors='lime', linewidths=1, linestyles='--', alpha=0.7)
            ax.set_title("Predicted Boundary (Red)\nvs Ground Truth (Green dashed)", fontsize=10)
            ax.axis('off')
            
            # Column 4: Error analysis
            ax = axes[sample_idx, 3]
            error_vis = np.zeros((IMG_SIZE, IMG_SIZE, 3))
            
            true_positives = pred_boundary & gt_boundary
            false_positives = pred_boundary & ~gt_boundary
            false_negatives = ~pred_boundary & gt_boundary
            
            error_vis[true_positives, 1] = 1.0  # Green
            error_vis[false_positives, 0] = 1.0  # Red
            error_vis[false_negatives, 2] = 1.0  # Blue
            
            ax.imshow(error_vis)
            
            # Calculate metrics
            if np.any(pred_boundary) and np.any(gt_boundary):
                tp = np.sum(true_positives)
                fp = np.sum(false_positives)
                fn = np.sum(false_negatives)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                ax.set_title(f"Error Analysis\nP: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}", fontsize=10)
            else:
                ax.set_title("Error Analysis", fontsize=10)
            
            ax.axis('off')
    
    plt.suptitle(f"ViT Autoencoder - Brain Tumor Boundary Detection Comparison", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(results_dir, f"{model_name}_boundary_comparison_grid.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison grid")

# ============================================================
# 1. VISION TRANSFORMER AUTOENCODER WITH CONTEXT DROPOUT
# ============================================================

class PatchEmbed(nn.Module):
    """
    Turn image into sequence of patch embeddings.
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 8 for 128/16
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)          # [B, embed_dim, H', W']
        x = x.flatten(2)          # [B, embed_dim, H'*W']
        x = x.transpose(1, 2)     # [B, N, embed_dim]
        return x


class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_chans=1,
        embed_dim=256,
        depth=4,
        num_heads=8,
        drop_rate=0.1,
        token_drop_rate=0.0  # <<< NEW
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(drop_rate)
        self.token_drop_rate = token_drop_rate

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=drop_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, 1, H, W]
        x = self.patch_embed(x)  # [B, N, D]
        x = x + self.pos_embed

        # --- NOVELTY: context dropout on patch tokens during training ---
        if self.training and self.token_drop_rate > 0.0:
            # keep probability = 1 - token_drop_rate
            keep_prob = 1.0 - self.token_drop_rate
            # mask shape [B, N, 1], Bernoulli(keep_prob)
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) < keep_prob).float()
            # zero-out dropped tokens
            x = x * mask

        x = self.dropout(x)
        x = self.transformer(x)  # [B, N, D]
        return x


class ViTDecoder(nn.Module):
    """
    Simple decoder: reshape tokens back into patch grid and deconvolve to full image.
    """
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        embed_dim=256,
        out_chans=1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.embed_dim = embed_dim

        # project tokens to a feature map
        self.proj = nn.Linear(embed_dim, embed_dim)

        # reshape to [B, D, H', W'] and deconv back to image
        self.deconv = nn.ConvTranspose2d(
            embed_dim,
            out_chans,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, N, D], N = H'*W'
        B, N, D = x.shape
        x = self.proj(x)                 # [B, N, D]
        x = x.transpose(1, 2)            # [B, D, N]
        x = x.view(B, D, self.grid_size, self.grid_size)  # [B, D, H', W']
        x = self.deconv(x)               # [B, C, H, W]
        # images are scaled to [-1, 1], so use tanh
        x = torch.tanh(x)
        return x


class ContextDroppedViTAE(nn.Module):
    """
    Transformer-based autoencoder for image reconstruction with context dropout.
    """
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_chans=1,
        embed_dim=256,
        depth=4,
        num_heads=8,
        drop_rate=0.1,
        token_drop_rate=0.0
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_rate=drop_rate,
            token_drop_rate=token_drop_rate
        )
        self.decoder = ViTDecoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_chans=in_chans
        )

    def forward(self, x):
        tokens = self.encoder(x)
        x_rec = self.decoder(tokens)
        return x_rec


# ============================================================
# 2. TRAIN / EVAL HELPERS
# ============================================================

def train_vit_ae(model, train_loader, optimizer, criterion, device, epochs, model_name="ViT_AE_CD"):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, _ in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}", leave=False):
            data = data.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"{model_name} - Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")


def evaluate_vit_ae(model, test_loader, device):
    """
    Returns:
        errors: np.array, shape (N,)
        labels: np.array, shape (N,)
    """
    model.eval()
    errors = []
    labels = []
    with torch.no_grad():
        for data, label in tqdm(test_loader, desc="Evaluating ViT_AE_CD", leave=False):
            data = data.to(device)
            output = model(data)
            # L1 reconstruction error per sample
            error = torch.mean(torch.abs(data - output), dim=(1, 2, 3))
            errors.extend(error.cpu().numpy())
            labels.extend(label.cpu().numpy())

    return np.array(errors), np.array(labels)


def plot_reconstruction_error(errors, labels, results_dir, model_name="vit_ae_cd"):
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

    if len(np.unique(labels)) < 2:
        print("⚠️ Warning: Only one class in labels, cannot compute ROC-AUC.")
        roc_auc = np.nan
        fpr, tpr, thresholds = np.array([]), np.array([]), np.array([])
    else:
        roc_auc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)

    np.save(os.path.join(results_dir, f"{model_name}_scores.npy"), scores)
    np.save(os.path.join(results_dir, f"{model_name}_labels.npy"), labels)
    np.save(os.path.join(results_dir, f"{model_name}_fpr.npy"), fpr)
    np.save(os.path.join(results_dir, f"{model_name}_tpr.npy"), tpr)
    np.save(os.path.join(results_dir, f"{model_name}_thresholds.npy"), thresholds)

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


# ============================================================
# 3. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Check for necessary imports
    try:
        import nibabel as nib
    except ImportError:
        print("Installing nibabel...")
        import subprocess
        subprocess.run(["pip", "install", "nibabel"])
        import nibabel as nib
    
    # Check for dataset
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"!!! WARNING: No patient folders found in '{DATA_PATH}'. Please extract or provide the correct BraTS dataset.")

    print(f"Using device: {DEVICE}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load BraTSDataset (assuming it's in AE.py)
    try:
        from AE import BraTSDataset
        print("✓ Loaded BraTSDataset from AE.py")
    except ImportError:
        print("Error: Could not import BraTSDataset from AE.py")
        print("Make sure AE.py is in the same directory or define BraTSDataset here.")
        exit(1)

    # --- DATASETS & LOADERS ---
    # Train only on normal slices for unsupervised anomaly detection
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
        raise ValueError("FATAL: Training dataset is empty. Check DATA_PATH and N_PATIENTS_TO_USE.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- MODEL, OPTIMIZER, LOSS ---
    model_name_vit_cd = "ViT_AE_CD"

    vit_ae_cd = ContextDroppedViTAE(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=1,
        embed_dim=Z_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        drop_rate=DROP_RATE,
        token_drop_rate=TOKEN_DROP_RATE  # <<< NOVELTY ACTIVE HERE
    ).to(DEVICE)

    optimizer = optim.Adam(vit_ae_cd.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # --- TRAIN ---
    print("\n" + "=" * 60)
    print("TRAINING ViT AUTOENCODER")
    print("=" * 60)
    train_vit_ae(vit_ae_cd, train_loader, optimizer, criterion, DEVICE, EPOCHS, model_name=model_name_vit_cd)

    # --- EVALUATE (reconstruction error as anomaly score) ---
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    errors_vit_cd, labels_vit_cd = evaluate_vit_ae(vit_ae_cd, test_loader, DEVICE)

    # --- PLOTS & METRICS ---
    print("\n" + "=" * 60)
    print("GENERATING BASIC METRICS")
    print("=" * 60)
    plot_reconstruction_error(errors_vit_cd, labels_vit_cd, RESULTS_DIR, model_name=model_name_vit_cd.lower())

    roc_auc_vit_cd = compute_and_save_metrics(
        scores=errors_vit_cd,
        labels=labels_vit_cd,
        model_name=model_name_vit_cd,
        results_dir=RESULTS_DIR,
        num_sample_outputs=100
    )

    # Save model weights
    torch.save(vit_ae_cd.state_dict(), os.path.join(RESULTS_DIR, f"{model_name_vit_cd}_model.pth"))
    print(f"✓ Saved model weights to {model_name_vit_cd}_model.pth")

    # --- BOUNDARY DETECTION VISUALIZATION ---
    print("\n" + "=" * 60)
    print("GENERATING BOUNDARY DETECTION VISUALIZATIONS")
    print("=" * 60)
    
    # Load sample brain images for visualization
    print("Loading sample brain images for visualization...")
    samples = load_sample_brain_images_for_visualization(num_samples=5)
    
    if samples:
        print(f"Loaded {len(samples)} sample images with tumors")
        
        # Generate individual boundary detection visualizations
        visualize_boundary_detection(vit_ae_cd, samples, RESULTS_DIR, model_name=model_name_vit_cd)
        
        # Generate comparison grid
        create_comparison_grid(samples, vit_ae_cd, RESULTS_DIR, model_name=model_name_vit_cd)
        
        # Generate additional visualization: Anomaly maps for all samples
        print("\nGenerating anomaly map comparison...")
        n_samples = len(samples)
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        
        if n_samples == 1:
            axes = np.array([axes])
        
        for sample_idx, sample in enumerate(samples):
            with torch.no_grad():
                img_tensor = sample['image'].to(DEVICE)
                reconstruction = vit_ae_cd(img_tensor)
                error = torch.abs(img_tensor - reconstruction)
                anomaly_map = error[0, 0].cpu().numpy()
                
                if anomaly_map.max() > anomaly_map.min():
                    anomaly_map_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
                else:
                    anomaly_map_normalized = anomaly_map
                
                # Original
                ax = axes[sample_idx, 0]
                ax.imshow(sample['original'], cmap='gray')
                ax.set_title(f"Sample {sample_idx + 1}: Original", fontsize=10)
                ax.axis('off')
                
                # Anomaly map
                ax = axes[sample_idx, 1]
                im = ax.imshow(anomaly_map_normalized, cmap='hot')
                ax.set_title("Anomaly Heatmap", fontsize=10)
                ax.axis('off')
                
                # Thresholded mask
                ax = axes[sample_idx, 2]
                threshold = np.percentile(anomaly_map_normalized, 95)
                tumor_mask = (anomaly_map_normalized > threshold).astype(np.float32)
                
                ax.imshow(sample['original'], cmap='gray')
                ax.imshow(tumor_mask, cmap='Reds', alpha=0.5)
                ax.set_title(f"Predicted Tumor (Threshold: {threshold:.2f})", fontsize=10)
                ax.axis('off')
        
        plt.suptitle("ViT Autoencoder - Anomaly Detection on Brain MRI", 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_path = os.path.join(RESULTS_DIR, f"{model_name_vit_cd}_anomaly_maps.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved anomaly maps to {model_name_vit_cd}_anomaly_maps.png")
        
        print(f"\n✓ Boundary detection visualizations complete!")
        print(f"  Check the following files in {RESULTS_DIR}:")
        print(f"  - {model_name_vit_cd}_boundary_sample_*.png")
        print(f"  - {model_name_vit_cd}_boundary_comparison_grid.png")
        print(f"  - {model_name_vit_cd}_anomaly_maps.png")
    else:
        print("✗ Could not load sample images for visualization")
    
    # Simple comparison CSV for this script
    df_results = pd.DataFrame([{"model": model_name_vit_cd, "roc_auc": float(roc_auc_vit_cd)}])
    df_results.to_csv(os.path.join(RESULTS_DIR, "model_comparison_roc_auc_vit_ae_cd_only.csv"), index=False)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(df_results)
    print(f"\nSaved ViT_AE_CD ROC-AUC table to: {os.path.join(RESULTS_DIR, 'model_comparison_roc_auc_vit_ae_cd_only.csv')}")
    
    print("\n" + "=" * 60)
    print("ViT AUTOENCODER TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nAll results saved to: {os.path.abspath(RESULTS_DIR)}")
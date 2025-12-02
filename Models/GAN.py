# ==========================
# GAN for Anomaly Detection
# ==========================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
import pandas as pd

# ======================================================================
# CONFIGURATION
# ======================================================================
DATA_PATH = '/kaggle/working/BraTS2021_Training_Data'  # change if needed
RESULTS_DIR = './results'
N_PATIENTS_TO_USE = 100
MODALITY = 't1ce'   # used inside BraTSDataset in your AE code
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 1          # increase for real experiments
Z_DIM = 128
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# NOTE: We assume BraTSDataset is already defined (from your AE code).
# from your_ae_file import BraTSDataset


# ======================================================================
# GENERATOR AND DISCRIMINATOR
# ======================================================================

class Generator(nn.Module):
    def __init__(self, z_dim, img_size):
        super(Generator, self).__init__()
        self.feature_side = img_size // 8  # 128 -> 16, consistent with conv downsamples
        self.fc1 = nn.Linear(z_dim, 128 * self.feature_side * self.feature_side)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = x.view(x.size(0), 128, self.feature_side, self.feature_side)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        # BraTSDataset scales images to [-1, 1], so tanh is appropriate
        x = torch.tanh(self.deconv3(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        feature_side = img_size // 8  # 128 -> 16
        self.fc1 = nn.Linear(128 * feature_side * feature_side, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))  # D(x) in [0,1]
        return x


# ======================================================================
# TRAINING FUNCTIONS
# ======================================================================

def train_gan(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, device, epochs):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        running_loss_g = 0.0
        running_loss_d = 0.0

        for real_data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            real_data = real_data.to(device)

            batch_size = real_data.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # -------------------------
            # Train Discriminator
            # -------------------------
            optimizer_d.zero_grad()

            # Real images
            out_real = discriminator(real_data)
            loss_d_real = criterion(out_real, real_labels)

            # Fake images
            z = torch.randn(batch_size, Z_DIM, device=device)
            fake_images = generator(z)
            out_fake = discriminator(fake_images.detach())
            loss_d_fake = criterion(out_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            # -------------------------
            # Train Generator
            # -------------------------
            optimizer_g.zero_grad()

            # Try to fool discriminator: want D(fake) ~ 1
            out_fake_for_g = discriminator(fake_images)
            loss_g = criterion(out_fake_for_g, real_labels)
            loss_g.backward()
            optimizer_g.step()

            running_loss_g += loss_g.item()
            running_loss_d += loss_d.item()

        print(f"Epoch [{epoch+1}/{epochs}] | Loss_G: {running_loss_g/len(train_loader):.4f} | "
              f"Loss_D: {running_loss_d/len(train_loader):.4f}")


# ======================================================================
# ANOMALY EVALUATION USING DISCRIMINATOR
# ======================================================================

def evaluate_gan_anomaly(discriminator, test_loader, device):
    """
    Use 1 - D(x) as anomaly score.
    Higher score => more anomalous.
    """
    discriminator.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for data, label in tqdm(test_loader, desc="Evaluating GAN for anomaly detection", leave=False):
            data = data.to(device)
            out = discriminator(data).view(-1)       # D(x) in [0,1]
            score = 1.0 - out                        # anomaly score
            scores.extend(score.cpu().numpy())
            labels.extend(label.cpu().numpy())

    return np.array(scores), np.array(labels)


def classify_anomalies(scores, threshold):
    """Binary predictions from continuous anomaly scores."""
    return (scores >= threshold).astype(int)


def plot_gan_anomaly_scores(scores, labels, results_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(scores[labels == 0], bins=50, alpha=0.7, label='Normal Slices', density=True)
    plt.hist(scores[labels == 1], bins=50, alpha=0.7, label='Anomalous Slices', density=True)
    plt.xlabel('GAN Anomaly Score (1 - D(x))')
    plt.ylabel('Density')
    plt.legend()
    plt.title('GAN Anomaly Score Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'gan_anomaly_score_distribution.png'))
    plt.close()


# ======================================================================
# ROC / AUC + SAMPLE OUTPUT SAVING (same pattern as AE/VAE)
# ======================================================================

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

    # Sample outputs (for thesis)
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


# ======================================================================
# CHOOSE THRESHOLD TO MAXIMIZE F1 (SO P/R/F1 NOT TRIVIALLY ZERO)
# ======================================================================

def find_best_threshold_by_f1(scores, labels):
    """
    Use precision-recall curve to find the threshold that maximizes F1.
    Returns: best_threshold, precision, recall, f1, tn, fp, fn, tp
    """
    # precision_recall_curve returns:
    #  precision, recall, thresholds (len(thresholds) = len(precision)-1)
    precision_arr, recall_arr, thr_arr = precision_recall_curve(labels, scores)

    # Compute F1 for each threshold (exclude last precision/recall element)
    f1_arr = 2 * precision_arr[:-1] * recall_arr[:-1] / (
        precision_arr[:-1] + recall_arr[:-1] + 1e-8
    )

    # If all F1 are zero, we still pick the best (but then the model truly has no separation)
    best_idx = np.argmax(f1_arr)
    best_threshold = thr_arr[best_idx]

    preds = (scores >= best_threshold).astype(int)

    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    return best_threshold, prec, rec, f1, tn, fp, fn, tp


# ======================================================================
# OPTIONAL: VISUALIZE REAL VS GENERATED IMAGES
# ======================================================================

def sample_real_and_fake_images(generator, test_loader, device, num_samples=1):
    generator.eval()
    real_batch, _ = next(iter(test_loader))
    real_batch = real_batch[:num_samples]
    with torch.no_grad():
        z = torch.randn(num_samples, Z_DIM, device=device)
        fake_batch = generator(z).cpu()
    return real_batch, fake_batch


def plot_generated_images(real_images, fake_images, results_dir, tag="final"):
    real_images = real_images.numpy()
    fake_images = fake_images.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(real_images[0, 0], cmap='gray')
    axes[0].set_title("Real")
    axes[0].axis('off')
    axes[1].imshow(fake_images[0, 0], cmap='gray')
    axes[1].set_title("Generated")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"gan_real_vs_fake_{tag}.png"))
    plt.close()


# ======================================================================
# MAIN EXECUTION
# ======================================================================

if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
    print(f"!!! WARNING: No patient folders found in '{DATA_PATH}'. "
          f"Please extract or provide the correct BraTS dataset.")

# ---- Datasets ----
# Train only on NORMAL slices (one-class training)
train_dataset = BraTSDataset(
    root_dir=DATA_PATH,
    num_patients=N_PATIENTS_TO_USE,
    img_size=IMG_SIZE,
    train=True,
    normal_only=True
)

# Test on both normal + abnormal slices
test_dataset = BraTSDataset(
    root_dir=DATA_PATH,
    num_patients=N_PATIENTS_TO_USE,
    img_size=IMG_SIZE,
    train=False,
    normal_only=False
)

if len(train_dataset) == 0:
    raise ValueError("Training dataset is empty. Check DATA_PATH and N_PATIENTS_TO_USE.")

print(f"Train slices: {len(train_dataset)}, Test slices: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---- Initialize models ----
generator = Generator(Z_DIM, IMG_SIZE).to(DEVICE)
discriminator = Discriminator(IMG_SIZE).to(DEVICE)
optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

# ---- Train GAN ----
train_gan(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, DEVICE, EPOCHS)

# ---- Evaluate GAN as anomaly detector ----
scores_gan, labels_gan = evaluate_gan_anomaly(discriminator, test_loader, DEVICE)

model_name_gan = "GAN"

# Compute ROC/AUC & save scores + ROC + sample outputs (for thesis)
roc_auc_gan = compute_and_save_metrics(
    scores=scores_gan,
    labels=labels_gan,
    model_name=model_name_gan,
    results_dir=RESULTS_DIR,
    num_sample_outputs=100
)

# ---- Choose threshold that MAXIMIZES F1 ----
best_thr, prec, rec, f1, tn, fp, fn, tp = find_best_threshold_by_f1(
    scores_gan, labels_gan
)
print(f"\n[GAN] Best threshold (by F1): {best_thr:.6f}")
print(f"[GAN] Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
print(f"[GAN] Confusion matrix (tn, fp, fn, tp): {tn}, {fp}, {fn}, {tp}")

# Save these metrics so you can merge with other models later
gan_metrics = pd.DataFrame(
    [{
        "model": "GAN",
        "prefix": "GAN",
        "roc_auc": float(roc_auc_gan),
        "best_threshold": float(best_thr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }]
)
gan_metrics.to_csv(os.path.join(RESULTS_DIR, "gan_metrics_best_f1.csv"), index=False)

# ---- Plots ----
plot_gan_anomaly_scores(scores_gan, labels_gan, RESULTS_DIR)

# Optional: visualize real vs generated images
real_vis, fake_vis = sample_real_and_fake_images(generator, test_loader, DEVICE, num_samples=1)
plot_generated_images(real_vis, fake_vis, RESULTS_DIR, tag="after_training")

# Save models
torch.save(generator.state_dict(), os.path.join(RESULTS_DIR, "gan_generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(RESULTS_DIR, "gan_discriminator.pth"))

# ---- Simple comparison CSV for this script ----
df_results = pd.DataFrame([{"model": model_name_gan, "roc_auc": float(roc_auc_gan)}])
df_results.to_csv(os.path.join(RESULTS_DIR, "model_comparison_roc_auc_gan_only.csv"), index=False)

print("\n=== Model Comparison (ROC-AUC, GAN only) ===")
print(df_results)
print(f"\nSaved GAN ROC-AUC table to: {os.path.join(RESULTS_DIR, 'model_comparison_roc_auc_gan_only.csv')}")

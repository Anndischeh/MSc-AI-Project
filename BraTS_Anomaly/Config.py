# config.py
import torch
import os

# Common configuration shared by all scripts
DATA_PATH = '/kaggle/working/BraTS2021_Training_Data'
RESULTS_DIR = './results'
N_PATIENTS_TO_USE = 100
MODALITY = 't1ce'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# datasets/brats.py
import os
import glob
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from config import MODALITY

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
        
        self.patient_dirs = train_dirs if self.train else val_dirs

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
            patient_id = os.path.basename(patient_dir)
            try:
                mod_path = glob.glob(os.path.join(patient_dir, f'*_{MODALITY}.nii.gz'))[0]
                seg_path = glob.glob(os.path.join(patient_dir, '*_seg.nii.gz'))[0]
            except IndexError:
                continue  # If files are missing, skip this patient
            
            mod_vol = nib.load(mod_path).get_fdata()
            seg_vol = nib.load(seg_path).get_fdata()

            for slice_idx in range(mod_vol.shape[2]):
                mod_slice = mod_vol[:, :, slice_idx]
                
                # Filtering logic (as in your scripts)
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
            slice_normalized = slice_resized * 0.0  # fallback
        
        slice_scaled = (slice_normalized * 2.0) - 1.0
        img_final = slice_scaled[np.newaxis, ...].astype(np.float32)

        return torch.from_numpy(img_final), torch.tensor(label, dtype=torch.long)

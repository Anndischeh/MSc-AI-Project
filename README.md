# Master Project: Unsupervised Anomaly Detection for Medical Imaging

**University:** City, University of London  
**Program:** MSc in Artificial Intelligence  
**Project Type:** Individual Master Project  

## Project Overview
This project focuses on developing unsupervised learning methods for detecting anomalies in medical imaging data. The aim is to leverage machine learning techniques to identify unusual patterns in medical scans without requiring labeled datasets.

## Contents
- `Master_Project_Proposal_Unsupervised_Anomaly_Detection_Medical_Imaging.pdf`: The project proposal.
- [Dataset link](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)

## Links
- [Project Proposal PDF](https://github.com/Anndischeh/MSc-AI-Project/blob/Proposal/Master_Project_Proposal_Unsupervised_Anomaly_Detection_Medical_Imaging.pdf)
---------------------------------------------------------------------------------------------------------
# Brain Tumor Anomaly Detection

## Overview

This repository contains a collection of deep learning models and utilities for detecting anomalies in brain tumor images using various architectures such as Autoencoders, Variational Autoencoders (VAE), Generative Adversarial Networks (GAN), GANomaly, and Vision Transformers (ViT). The models are evaluated on the BraTS 2021 dataset, which consists of brain MRI scans annotated for tumor regions.

### Supported Models

1. **Autoencoder (AE)**: A basic autoencoder model used for reconstructing MRI images and identifying anomalies based on reconstruction errors.
2. **Variational Autoencoder (VAE)**: A probabilistic model for generating MRI images and detecting anomalies.
3. **Generative Adversarial Network (GAN)**: Used for generating realistic images and detecting anomalies by comparing real vs. generated images.
4. **GANomaly**: Anomaly detection using GANs, where the generator reconstructs input images and the discriminator classifies the images as normal or anomalous.
5. **Vision Transformer (ViT)**: A transformer-based model adapted for image anomaly detection by processing image patches.

## Project Structure

```

├── /Models
│   ├── Autoencoder.py        # Autoencoder architecture
│   ├── GAN.py                # GAN architecture
│   ├── Ganomaly.py           # GANomaly model for anomaly detection
│   ├── VariationalAutoencoder.py  # VAE model
│   └── Vit.py                # Vision Transformer for anomaly detection
│
├── /Dataset
│   └── BraTS_dataset.py  # The BraTS 2021 dataset (MRI brain scans)
│
├── /results
│   └── output_images          # Directory where the model results and visualizations are saved
│
├── Main.py                   # Entry point for training and evaluating models             
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

To set up the environment, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/Anndischeh/MSc-AI-Project.git
   cd brain-tumor-anomaly-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download and place the BraTS 2021 dataset in the `/data` directory. The dataset should consist of the MRI scans and the corresponding segmentation masks.

## Usage

### Training the Models

To train any of the models (Autoencoder, VAE, GAN, GANomaly, ViT), run the following script. This will train the selected model and save the results in the `/results` directory:

```bash
python scripts/Main.py --model [autoencoder|vae|gan|ganomaly|vit] --epochs 100 --batch_size 32
```

You can change the model and adjust hyperparameters such as epochs and batch size.

### Model Evaluation

After training, the models can be evaluated using a script that calculates various performance metrics (e.g., ROC-AUC, Precision, Recall, F1-Score). You can visualize the evaluation results (e.g., confusion matrices, ROC curves, anomaly score distributions) with the following script:

```bash
python scripts/run_comparison_fixed.py
```

### Visualization

After running the evaluation, visualizations of the model's performance are saved in the `/results` directory. This includes:

* **ROC Curves** for each model
* **Confusion Matrices** showing the classification results for each model
* **Anomaly Score Distribution** visualizing how well the model detects anomalies
* **Boundary Detection Visualizations** using ViT (Vision Transformer) for detecting tumor boundaries

## Models Description

### Autoencoder (AE)

The Autoencoder model learns to compress (encode) MRI images into a lower-dimensional representation and then reconstruct them. Anomalies are detected by measuring reconstruction errors. Higher reconstruction errors indicate abnormal regions (e.g., tumors).

### Variational Autoencoder (VAE)

The VAE is a probabilistic extension of the autoencoder that learns to encode the images into a latent space following a Gaussian distribution. Anomalies are detected by comparing the reconstruction error and the latent space distribution.

### GAN (Generative Adversarial Network)

The GAN consists of two neural networks: the generator (which creates fake images) and the discriminator (which distinguishes real from fake images). In this project, GAN is used to generate images and detect anomalies by comparing the generated images with real ones.

### GANomaly

GANomaly uses GANs for anomaly detection. The generator creates a reconstruction of the image, and the discriminator classifies it as either "normal" or "anomalous" based on the reconstruction error.

### Vision Transformer (ViT)

The ViT model divides the input images into non-overlapping patches, processes them through transformer layers, and reconstructs the image. Anomalies are detected by comparing the reconstructed images with the original ones. Additionally, ViT helps in boundary detection for tumor regions.

## Requirements

* Python 3.x
* PyTorch 2.0 or higher
* Required libraries:

  * numpy
  * pandas
  * matplotlib
  * scikit-learn
  * torch
  * torchvision
  * tqdm
  * opencv-python
  * nibabel

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



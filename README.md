# Master Project
## Unsupervised Anomaly Detection for Medical Imaging
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)
![Domain](https://img.shields.io/badge/domain-medical%20imaging-red)
![Learning](https://img.shields.io/badge/learning-unsupervised%20anomaly%20detection-purple)
![License](https://img.shields.io/badge/license-MIT-green)
![Research](https://img.shields.io/badge/type-research-blueviolet)
![Dataset](https://img.shields.io/badge/dataset-BraTS%202021-yellow)


**University:** City, University of London  
**Program:** MSc in Artificial Intelligence  
**Project Type:** Individual Master Project  

## Executive Summary
This project investigates reconstruction-based unsupervised anomaly detection for brain MRI,
addressing the challenge of limited labeled data in clinical environments.

Five architectures (AE, VAE, GAN, GANomaly, and a proposed ViT-based Autoencoder with Context Dropout)
were implemented and benchmarked on [the BraTS 2021 dataset](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1).

The proposed ViT-based model achieved the strongest performance (ROC-AUC: 0.862, AP: 0.838),
demonstrating the effectiveness of global context modelling for tumour detection and localisation.

## Problem Definition
- Manual annotation of medical imaging data is expensive and time-consuming
- Supervised models struggle to generalise with limited labeled data
- The goal is to detect and localise anomalies using only healthy samples during training

## Design Choices
- Reconstruction-based anomaly detection was selected to avoid dependency on labeled anomalies
- Multiple architectures were benchmarked to avoid a one-size-fits-all approach
- Vision Transformers were explored to capture long-range spatial dependencies
- Context Dropout was introduced to improve robustness to local noise

## Key Results
- ViT_AE_CD achieved the highest performance:
  - ROC-AUC: 0.862
  - Average Precision: 0.838
- Transformer-based models outperformed CNN and GAN-based baselines
- Anomaly heatmaps showed improved localisation of tumour regions

## Engineering & Reproducibility
- Modular model implementations
- Configuration-driven experiments
- Deterministic evaluation pipelines
- Version-controlled experiments (Git)

**Tech Stack:** Python, PyTorch, Torchvision, NumPy, scikit-learn, OpenCV, NiBabel


## Notes on Production Deployment
While this project is research-focused, the pipeline could be extended to production by:
- Adding data validation and drift monitoring
- Introducing model performance tracking post-deployment
- Optimising inference latency for clinical workflows

# Brain Tumor Anomaly Detection

## Overview

This repository contains a collection of deep learning models and utilities for detecting anomalies in brain tumor images using various architectures such as Autoencoders, Variational Autoencoders (VAE), Generative Adversarial Networks (GAN), GANomaly, and Vision Transformers (ViT). The models are evaluated on the BraTS 2021 dataset, which consists of brain MRI scans annotated for tumor regions.


## Contents
- `Master_Project_Proposal_Unsupervised_Anomaly_Detection_Medical_Imaging.pdf`: The project proposal.
- [Dataset link](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)

## Links
- [Project Proposal PDF](https://github.com/Anndischeh/MSc-AI-Project/blob/Proposal/Master_Project_Proposal_Unsupervised_Anomaly_Detection_Medical_Imaging.pdf)
---------------------------------------------------------------------------------------------------------

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






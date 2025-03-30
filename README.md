# Generative EEG Augmentation and Source Localization via GANs, VAEs, and Simulation-Based Inference

## Overview
This project was done for the course of Generative Neural Networks for the sciences and explores advanced generative and inference techniques for EEG data augmentation and source localization.
Our goal is to overcome challenges of limited EEG datasets by using deep generative models-specifically Conditional Wasserstein GANs (WGAN-GP)
and Variational Autoencoders (VAEs)—to produce realistic synthetic EEG signals. These synthetic data are used to improve the performance of downstream EEG classification tasks and to provide a proof-of-concept for
simulation-based inference in EEG source localization.

The Current Project modules are as follows

- Data Preprocessing: We preprocess raw EEG data using MNE-Python, applying filtering, artifact removal, epoching, and normalization.
- Generative Modeling: We implement a conditional WGAN-GP to learn and generate EEG epochs conditioned on binary emotion labels (e.g., “positive” vs. “negative”). We also develop a VAE for comparison.

FuTure work to be done
- EEG Classification
- Source Localization

## Research Questions
1. EEG Generation Quality: Can deep generative models produce realistic EEG signals in both the time and frequency domains?
2.	Class Conditioning: Does conditioning on class labels allow generation of EEG epochs that capture class-specific patterns (e.g., “positive” vs. “negative” emotion)?
3.	Source Localization: Can simulation-based deep learning accurately localize neural sources from EEG, and how does it compare with classical methods?


### Repository structure


### Installation

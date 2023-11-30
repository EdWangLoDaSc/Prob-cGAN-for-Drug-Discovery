# Prob-cGAN-for-Drug-Discovery
A novel probabiliatic conditional GAN method to solve the Drug Activity question

##Introduction
This repository contains the implementation of a novel probabilistic conditional Generative Adversarial Network (cGAN) method aimed at addressing the Drug Activity question in the field of Drug Discovery. The project demonstrates how advanced machine learning techniques can be applied to the complex and critical domain of pharmaceuticals.

##Features
Probabilistic Conditional GAN: An innovative approach in the application of GANs to model the uncertainty in drug discovery.
Preprocessing and Data Handling: Efficient handling of diverse datasets related to drug compounds and their activities.
Evaluation Metrics: Rigorous evaluation methods to assess the performance and reliability of the generated models.
Visualization Tools: Integrated visualization tools for in-depth analysis of the GAN outputs and training process.

##Repository Structure
Baseline/: Baseline models for comparison.
models/: Core cGAN model implementations.
nn_spec/, util_spec/: Specific neural network configurations and utilities.
constants.py, dataset_list.py, utils.py: Utility scripts for constants, dataset handling, and general-purpose functions.
preprocess_dataset.py: Script for dataset preprocessing.
evaluation.py: Evaluation metrics and functions.
main.py: Main script to run the cGAN models.
epoch_best.pt: Saved model state for the best epoch.
requirements.txt: List of dependencies.
visualization.py: Visualization tools and scripts.
Standard cGAN.ipynb, Word Embeding Example.ipynb: Jupyter notebooks for demonstrative purposes.

# BrainTumorNet
BrainTumorNet: A Lightweight CNN Model for Brain Tumor Segmentation
## Overview
BrainTumorNet is a lightweight Convolutional Neural Network (CNN) designed for the segmentation of brain tumors in MRI images. This model offers a streamlined approach to identify and delineate tumor regions accurately, reducing computational complexity while maintaining high segmentation performance.

## Features
- Lightweight Architecture: Optimized CNN model with fewer layers for reduced computational overhead.
- Attention Mechanism: Incorporates an attention mechanism to highlight important regions, enhancing segmentation accuracy.
- Custom Loss Function: Combines Binary Cross-Entropy (BCE) and Dice loss to ensure both pixel-wise accuracy and spatial overlap.
- Data Augmentation: Employs various augmentation techniques to improve model generalization.
- Meta-Heuristic Optimization: Utilizes Whale Optimization Algorithm (WOA), Bee Algorithm, and Firefly Algorithm for hyperparameter tuning.
## Dataset
The model is trained and evaluated on the LGG Segmentation Dataset by Mateusz Buda and Maciej A. Mazurowski, consisting of 7,858 .tif files. The dataset includes MRI images of lower-grade gliomas and corresponding highly accurate segmented masks.

## Methodology
### Data Preprocessing
- Dataset Analysis: Visualization of tumor and non-tumor distribution.
- Normalization: Calculation of mean and standard deviation for dataset normalization.
- Augmentation: Application of rotations, flips, and other transformations to enhance dataset diversity.
### Model Architecture
- Convolutional Layers: Progressive increase in feature depth with convolutional layers.
- Pooling and Upsampling: Max pooling for dimensionality reduction and bilinear interpolation for upsampling.
- Attention Mechanism: Enhances feature maps to focus on tumor regions.
- Sigmoid Activation: Final activation for producing a probability map for segmentation.
### Training and Optimization
- Custom Loss Function: Combines BCE and Dice loss for optimal training.
- Optimization Algorithms: Hyperparameters tuned using WOA, Bee Algorithm, and Firefly Algorithm.
### Evaluation and Visualization
- Metrics: Calculation of accuracy, precision, recall, F1 score, IoU, and Dice coefficient.
- Visualization: Display of original images, ground truth masks, and model predictions for qualitative assessment.
### Results
Accuracy: 99.70%
Precision: 90.08%
Recall: 93.85%
F1 Score: 91.92%
IoU: 85.06%
Dice Coefficient: 91.92%

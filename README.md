# Polymorpho-tools
Analysis codes for quantifying polymorphonuclear nuclear shape transitions

## Code Summary

This repository contains a collection of Python scripts for analyzing 3D microscopy images of cell nuclei, with a focus on quantifying nuclear lobes and classifying nuclear morphology.

### Core Analysis

*   **`hierarchical.py`**: The main script for lobe counting using a hierarchical clustering approach. It iteratively erodes a 3D binary image of a nucleus, assigning unique string identifiers to each resulting component based on its lineage. A custom distance metric based on the longest common prefix of these strings is used to build a distance matrix. Finally, hierarchical clustering is performed on this matrix, and the resulting dendrogram is used to estimate the number of lobes.

### Feature Extraction and Machine Learning

*   **`features.py`**: Extracts a variety of morphological and textural features from the 3D nuclear images. These features include volume, surface area, sphericity, and Haralick texture features. It then uses Principal Component Analysis (PCA) for dimensionality reduction and applies clustering algorithms like K-Means and HDBSCAN to group nuclei based on their features.

*   **`labeled_tests.py`**: A script for supervised machine learning classification. It extracts a comprehensive set of features (volume, sphericity, Haralick features, Zernike moments) and uses a Support Vector Classifier (SVC) with cross-validation to predict the number of lobes and the presence of holes based on these features.

*   **`centroid_distance.py`**: Implements a lobe counting method based on signal processing. It calculates the 3D mesh of the nucleus, finds its centroid, and then generates a 1D signal representing the maximum distance from the centroid to the surface as a function of the azimuthal angle. Peaks in this signal are counted to estimate the number of lobes. It also trains an SVC on the generated signals.

### Experimental and Utility Scripts

*   **`scikit_test.py`**: An experimental script for exploring different recursive erosion strategies to count nuclear components. It contains several approaches and a significant amount of commented-out code, suggesting it was used as a scratchpad for developing the methods in `hierarchical.py`.

*   **`downsampling_test.py`**: Analyzes the effect of image quantization (reducing the number of gray levels) on feature extraction. It compares different quantization strategies (K-Means vs. uniform) and measures the impact on Mean Squared Error (MSE) and the time taken to compute Haralick features.

*   **`2D_test.py`**: An exploratory script for applying lobe counting algorithms to 2D images, likely from Geimsa-stained samples. It contains functions for iterative erosion and counting in a 2D context.

*   **`zernike.py`**: A focused script for experimenting with the extraction of Zernike moments from different 2D slices of the 3D nucleus.

*   **`volume_check.py`**: A utility script to calculate and visualize the distribution of nuclear volumes across a dataset of images.

*   **`import_torch.py`**: A basic script for setting up and testing a PyTorch environment. It demonstrates loading a pre-trained ResNet model and working with the CIFAR-10 dataset, likely for future deep learning applications.

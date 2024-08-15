# Distributed and Secure Kernel-Based Quantum Machine Learning

This repository contains the implementation of the algorithms presented in the paper titled **"Distributed and Secure Kernel-Based Quantum Machine Learning"**. It provides all the necessary code to reproduce the experiments described in the paper, including dataset loading, centralized and distributed computations, and experiment results.

## Repository Structure

- **`library.py`**: This file contains all the core functions required to implement both the centralized and the distributed and secure kernel-based quantum machine learning algorithm as described in the paper. It serves as the backbone of the implementation, handling tasks such as kernel computation, noise management, and distributed processing.

- **Datasets Folders**: Each folder in the repository corresponds to a specific dataset used in the experiments. The folders include:
  - **Data Loading Scripts**: Scripts to load and preprocess the respective datasets.
  - **Distributed QML Scripts**: Code to run the distributed kernel-based quantum machine learning algorithm on the dataset.
  - **Experimental Results**: Results obtained from running the algorithm on the dataset, including kernel matrices and performance metrics.

# Distributed and Secure Kernel-Based Quantum Machine Learning

This repository contains the implementation of the algorithms presented in the paper titled **["Distributed and Secure Kernel-Based Quantum Machine Learning"](https://openreview.net/forum?id=3jdI0aEW3k)**. It provides all the necessary code to reproduce the experiments described in the paper, including dataset loading, centralized and distributed computations, and experiment results.

## Repository Structure

- **`library.py`**: This file contains all the core functions required to implement both the centralized and the distributed and secure kernel-based quantum machine learning algorithm as described in the paper. It serves as the backbone of the implementation, handling tasks such as kernel computation, noise management, and distributed processing.

- **Datasets Folders**: Each folder in the repository corresponds to a specific dataset used in the experiments. The folders include:
  - **Data Loading Scripts**: Scripts to load and preprocess the respective datasets.
  - **Distributed QML Scripts**: Code to run the distributed kernel-based quantum machine learning algorithm on the dataset.
  - **Experimental Results**: Results obtained from running the algorithm on the dataset, including kernel matrices and performance metrics.

- **other_kernels**: This folder contains the code for implementing the polynomial, RBF, and Laplacian kernels as proposed in the paper.

## Citation

Please consider citing our work if it is beneficial to your research. 
- **Paper**
```bibtex
@article{
swaminathan2025distributed,
title={Distributed and Secure Kernel-Based Quantum Machine Learning},
author={Arjhun Swaminathan and Mete Akg{\"u}n},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=3jdI0aEW3k},
note={}
}
```
- **Code**
```bibtex
@software{Swaminathan2025QML,
  author    = {Swaminathan, A. and Akg{\"u}n, M.},
  title     = {Distributed and Secure Kernel-Based Quantum Machine Learning - code},
  version = {v1.0},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18743937},
  url       = {https://doi.org/10.5281/zenodo.18743937}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact for Questions
`arjhun.swaminathan@uni-tuebingen.de`, `mete.akguen@uni-tuebingen.de`

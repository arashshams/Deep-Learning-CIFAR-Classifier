# Deep Learning CIFAR Classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/deep-learning-cifar-classifier/blob/main/notebooks/cifar10_image_classification.ipynb)

Deep learning image classification project using the CIFAR-10 dataset (with planned extensions to CIFAR-100).  
This project compares several convolutional neural network (CNN) architectures and a transfer learning model (VGG16), and will be extended with explainability (Grad-CAM), data augmentation, and an interactive prediction interface.

## Repository structure

```text
deep-learning-cifar-classifier/
├── notebooks/
│   └── cifar10_image_classification.ipynb   # main notebook (training, evaluation, experiments)
├── src/                                     # python modules (to be added)
├── models/                                  # saved models (not versioned by default)
├── reports/
│   └── figures/                             # plots and figures
├── requirements.txt                         # python dependencies
└── .gitignore

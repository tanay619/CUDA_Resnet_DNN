## Accelerated CIFAR-10 Image Classification

A ResNet-based image classification model for the CIFAR-10 dataset, leveraging CUDA GPU acceleration for efficient training and evaluation, with seamless CPU fallback.

## Code Organization

```data/```
## CIFAR-10 Dataset Description

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

The dataset is divided into five training batches and one test batch, each containing 10,000 images. The test batch has exactly 1,000 randomly-selected images from each class. The training batches contain the remaining images in random order, with each class appearing exactly 5,000 times across the batches.

```src/```
The source code should be placed here in a hierarchical fashion, as appropriate.

Explanation of the `resnet_cifar.py` Script

The provided `resnet_cifar.py` script is designed to train and evaluate a ResNet model on the CIFAR-10 dataset. The script supports both CPU and GPU execution, determined by the command-line argument provided (`F` for CPU and `T` for GPU).

1. **Dataset Preparation**:
   - The script downloads and prepares the CIFAR-10 dataset using the `torchvision` library.
   - The `ImageDataset` class is defined to handle different data splits (`train`, `val`, `test`).

2. **Model Definition**:
   - A ResNet model (`Resnet_Q1`) is defined using PyTorch's `nn.Module`.
   - The model includes several convolutional layers, batch normalization, ReLU activations, and a fully connected layer.

3. **Training, Validation, and Evaluation**:
   - The script includes `trainer`, `validator`, and `evaluator` functions to handle training, validation, and testing of the model.
   - During training, the model's state is saved to a checkpoint file (`checkpoint.pth`) after each epoch.
   - The evaluator function loads the model's state from the checkpoint file to evaluate the model on the test set.

By following the instructions in the `INSTALL` file and using the `Makefile`, you can easily set up and run the project on both CPU and GPU, with outputs saved to `output.txt`.

---

You can now create the `Makefile`, `INSTALL` file, and update your `README.md` with the provided content. This will ensure your project is well-documented and easy to use.



```README.md```
This file holds the description of the project to help anyone cloning or deciding whether to clone this repository to understand its purpose.

```INSTALL```
This file contains the human-readable set of instructions for installing the code so it can be executed. If possible, it should be organized around different operating systems to accommodate various constraints.

```Makefile```
This file contains scripts for building and running your project's code in an automatic fashion. It includes commands to run the project on both CPU and GPU and to save the output to output.txt.

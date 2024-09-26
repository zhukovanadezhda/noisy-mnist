# 🖼️ Noisy MNIST Handwritten Digit Classification

This project involves classifying images of handwritten digits from the MNIST dataset, with added noise such as rotations, shifts, zooms, and lighting variations. The dataset includes 60,000 images for training and 10,000 images for testing, all in 28×28 pixel grayscale format. The goal is to develop models capable of accurately recognizing noisy digits. We explore two model architectures:

1. **Fully Connected Neural Networks (FCNN)**.
2. **Convolutional Neural Networks (CNN)**.

## 📊 Dataset

The dataset is derived from the MNIST handwritten digits and includes different modifications. It can be generated by:

   ```bash
  cd data
  python ../src create_noisy_digits.py
   ```

## 🔄Installation

### Clone the repository

```bash
git clone git@github.com:zhukovanadezhda/noisy-mnist.git
cd noisy-mnist
```
### Setup the conda environment

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html). Create the `deep-learning` conda environment:

```bash
conda env create -f environment.yml
```

### Load the environment

```bash
conda activate deep-learning
```

> 💡**Note:** To deactivate an active environment, use:
> ```bash
> conda deactivate
> ```

   
## 📄 References

Deng, L. (2012). The mnist database of handwritten digit images for machine learning research. *IEEE Signal Processing Magazine*, 29(6), 141–142.

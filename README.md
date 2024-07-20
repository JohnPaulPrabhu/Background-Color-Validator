# Image-Blur-Detector

This project focuses on detecting whether a given image is blurry or not using computer vision techniques.


https://github.com/user-attachments/assets/89a928a6-cb13-4127-b697-72f77dfb7f1d



## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)

## Introduction

The Image Blur Detector is a tool designed to identify and classify blur in images. It uses various image processing techniques and deep learning models to achieve high accuracy in blur detection and classification.

## Installation

To get started with the project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/JohnPaulPrabhu/Image-Blur-Detector.git
cd Image-Blur-Detector
pip install -r requirements.txt
```

## Usage

To use the Image Blur Detector, first run the final_training.py to create the model. After that, change the output path to your desired video path then you can run the following command for the inference:

```bash
python main.py
```

The script will output whether the image is blurry or not.

## Methodology

The methodology used for detecting and classifying blur involves several steps:

1. **Data Loading:** Images are loaded and preprocessed using the `data_loading.py` script.
2. **Model Training:** The deep learning model is trained using the `training.py` and `final_training.py` scripts.
3. **Hyperparameter Optimization:** The `hyperparameter_optimization.py` script is used to fine-tune the model's parameters.
4. **Evaluation:** The `evaluation.py` script evaluates the model's performance on test data.

## Results

The Image Blur Detector has been tested on a variety of images and has shown high accuracy in detecting and classifying blur. Sample result is shown above.

---

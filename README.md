# Satellite Land Cover Classification Using ANN Fusion Model 🌍

## Overview

This project presents a satellite image classification system for identifying different land-cover categories using an Artificial Neural Network (ANN) combined with feature fusion techniques.

The main objective is to investigate the impact of combining different image feature representations and evaluate whether feature fusion can improve classification performance compared to using individual feature extraction methods.

The proposed approach combines:

- Color-based features using HSV Color Histograms.
- Texture-based features using Local Binary Pattern (LBP).
- An Artificial Neural Network (ANN) classifier for final classification.

By integrating complementary visual information, the model aims to achieve a richer representation of satellite images and improve land-cover classification accuracy.


---

# Dataset

## EuroSAT Dataset

The project uses the EuroSAT dataset, a benchmark dataset for satellite image classification.

Dataset characteristics:

- Total images: **27,000 satellite images**
- Number of classes: **10 land-cover categories**
- Image type: RGB satellite images


## Land Cover Classes

The dataset contains the following categories:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake


## Dataset Split

The dataset was divided into three subsets:

| Dataset | Number of Images | Percentage |
|---|---|---|
| Training | 18,900 | 70% |
| Validation | 5,400 | 20% |
| Test | 2,700 | 10% |


## Dataset Exploration

Exploratory data analysis was performed to:

- Analyze dataset distribution.
- Verify class balance.
- Visualize sample images from each category.
- Understand visual similarities and differences between land-cover classes.

(Add dataset exploration image here)


---

# Feature Extraction & Fusion Approach

## Color Feature Extraction

### HSV Color Histogram

Color information was extracted using HSV color histograms.

The purpose of using color features is to capture spectral characteristics of different land-cover types, such as:

- Vegetation regions.
- Water bodies.
- Urban areas.

HSV representation helps separate color information from intensity, making it suitable for satellite image analysis.


---

## Texture Feature Extraction

### Local Binary Pattern (LBP)

Texture information was extracted using Local Binary Pattern (LBP).

LBP captures local spatial patterns and texture variations within satellite images.

This helps differentiate between visually similar classes based on surface structures and patterns.


---

## Feature Fusion

The extracted color and texture features were combined into a single feature vector.

The fusion process allows the model to utilize complementary information:

- Color features describe spectral characteristics.
- LBP features describe spatial texture patterns.

Combining both representations provides a more informative feature space compared to using a single feature extraction method.


---

# Model Architecture

The fused feature vector was used as input to an Artificial Neural Network (ANN) classifier.

Pipeline:

Satellite Image

    ↓


Feature Extraction

    ↓


HSV Color Features + LBP Texture Features

    ↓


Feature Fusion

    ↓


Feature Scaling

    ↓


Artificial Neural Network (ANN)

    ↓


Land Cover Classification



Model details:

- Model type: Artificial Neural Network
- Input: Fused feature vector
- Output: Land-cover class probabilities
- Activation: Classification output layer
- Evaluation: Accuracy, Precision, Recall, and F1-score


---

# Experimental Results

Different feature extraction strategies were evaluated to analyze the effect of feature fusion.


| Approach | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| ANN + LBP | 77.3% | 76.2% | 76.5% | 76.3% |
| ANN + Color Features | 78.3% | 77.6% | 77.6% | 77.2% |
| ANN + Color + LBP Fusion | **89.6%** | **89.1%** | **89.0%** | **89.0%** |


## Results Analysis

The fusion-based approach achieved the highest performance compared to using individual feature representations.

The improvement demonstrates that combining color and texture information provides a more complete representation of satellite images.

The ANN classifier benefits from:

- Color-based information to identify different surface characteristics.
- Texture-based information to distinguish spatial patterns.
- A combined feature space that improves class separation.


---

# Demo

An interactive Streamlit application was developed to test the trained model.

The demo allows users to:

- Upload a satellite image.
- Select example satellite images.
- Predict the land-cover category.
- Display the confidence score.


(Add demo screenshot here)


---

# How to Run

Clone the repository:

```bash
git clone <repository-url>

cd LandCover

Install dependencies:

bash
pip install -r requirements.txt

Run the Streamlit demo:

bash
cd demo

streamlit run app.py

Project Structure
plaintext
LandCover/
│
├── demo/
│   ├── app.py
│   └── requirements.txt
│
├── model/
│   ├── ann_fusion_model.h5
│   ├── fusion_scaler.pkl
│   └── fusion_label_encoder.pkl
│
├── samples/
│   ├── forest.jpg
│   ├── river.jpg
│   └── seaLake.jpg
│
├── notebooks/
│   └── ANN_+_Fusion.ipynb
│
├── requirements.txt
└── README.md

Technologies
- Python
- TensorFlow / Keras
- OpenCV
- Scikit-image
- Scikit-learn
- Streamlit
- NumPy
- Matplotlib

Future Improvements
- Compare the fusion approach with CNN-based feature extraction.
- Experiment with pretrained deep learning models.
- Improve generalization using data augmentation.
- Deploy the model as a cloud-based application.

Author
Rana Alzahrani
Computer Science Graduate | Artificial Intelligence
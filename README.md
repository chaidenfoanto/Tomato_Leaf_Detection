# 🍅 Tomato Leaf Disease Detection (CNN + OpenCV)

## 📌 Overview

This project is a **real-time tomato leaf disease detection system** using **Deep Learning (CNN)** and **Computer Vision (OpenCV)**. The system captures live video from a webcam, detects whether a tomato leaf is present, and classifies the type of disease with confidence scoring and treatment suggestions.

---

## 🧠 Machine Learning Approach

This project uses **Supervised Learning**, specifically:
* **Task**: Multiclass Image Classification
* **Model**: Convolutional Neural Network (CNN)
* **Framework**: TensorFlow / Keras

The model is trained on labeled tomato leaf images to classify multiple disease categories.

---

## 🔄 System Pipeline

```text
Webcam Input
     ↓
Leaf Detection (HSV Color Filter)
     ↓
Crop & Preprocess (Resize 224x224, Normalize)
     ↓
CNN Model Prediction
     ↓
Confidence Filtering
     ↓
Display Result (Disease + Suggestion)
```

---

## 🚀 Features

* Real-time detection via webcam
* Leaf detection using HSV color segmentation
* CNN-based disease classification
* Confidence threshold filtering
* Informative output:

  * Disease name
  * Confidence score
  * Description
  * Treatment suggestion

---

## 📊 Model Performance

### Classification Report

| Class                  | Precision | Recall | F1-Score |
| ---------------------- | --------- | ------ | -------- |
| bacterial_spot         | 0.87      | 0.92   | 0.90     |
| early_blight           | 0.73      | 0.83   | 0.78     |
| healthy                | 0.54      | 1.00   | 0.70     |
| late_blight            | 0.96      | 0.71   | 0.82     |
| leaf_mold              | 0.98      | 0.71   | 0.83     |
| septoria_leaf_spot     | 0.77      | 0.83   | 0.80     |
| spotted_spider_mite    | 0.93      | 0.60   | 0.73     |
| target_spot            | 0.76      | 0.69   | 0.72     |
| yellow_leaf_curl_virus | 1.00      | 0.84   | 0.92     |

### Overall Performance

* **Accuracy:** 79%
* **Macro F1-score:** 0.80
* **Weighted F1-score:** 0.80

---

## 🔍 Model Insight

* High performance on:
  * bacterial_spot
  * yellow_leaf_curl_virus
* Lower performance on:
  * healthy (low precision)
  * spider_mite (low recall)

### Possible Reasons:

* Class imbalance
* Visual similarity between diseases
* Real-time noise (lighting, blur)

---

## 📂 Dataset

Dataset used:
🔗 https://www.kaggle.com/datasets/muhammadmasdar/tomato-disease-ready

Additional access:
🔗 https://drive.google.com/drive/folders/1jBkckCgrZcV9nb1As1whHf5c6Cdl8Doj?usp=sharing

### Notes

* Dataset is not included in this repository
* Please download manually

---

## 🤖 Model

Stored using Git LFS:

```bash
model_cnn_penyakit_daun.h5
```

---

## 🛠️ Requirements

```bash
pip install tensorflow numpy opencv-python
```

---

## ▶️ How to Run

### 1. Clone Repository

```bash
git clone https://github.com/chaidenfoanto/Tomato_Leaf_Detection.git
cd Tomato_Leaf_Detection
```

---

### 2. Run Application

```bash
python main.py
```

---

### 3. Usage

* The camera will turn on automatically  
* Point the camera at a tomato leaf  
* Press **Q** to exit

---

## 🔁 Reproducibility (Important)

To reproduce results:

1. Download dataset
2. Train model using your training notebook/script
3. Save model as:

```bash
model_cnn_penyakit_daun.h5
```

---

## 🌱 Real-World Use Case

* Early detection of plant diseases
* Helping farmers monitor crop health
* Educational tool for agriculture students

---

## ⚠️ Limitations

* Sensitive to lighting conditions
* Not robust for blurred images
* Only trained for tomato leaves

---

## 🎯 Future Improvements

* Mobile app integration (Flutter)
* Convert model to TensorFlow Lite
* Improve dataset balance
* Add image upload feature

---

## Acknowledgment

Dataset provided by Kaggle community.

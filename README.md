# 🚦 GTSRB Traffic Sign Classification with Deep Learning (Keras + CNN)

A comprehensive computer vision project that classifies traffic signs using a Convolutional Neural Network (CNN) trained on the [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). This project includes detailed preprocessing, data augmentation, model tuning, evaluation, and visualization steps.

---

## 🧠 Overview

This project aims to classify traffic signs into 43 categories using a deep learning model built with Keras and TensorFlow. It leverages the GTSRB dataset, applies a robust image preprocessing pipeline, uses data augmentation to improve generalization, and evaluates performance on unseen test images.

---

## 📁 Folder Structure

```
.
├── Traffic_Sign_Classification.ipynb       # Main notebook (with explanations and markdown)
├── GTSRB_dataset/
│   ├── train.p                    # Pickled training data
│   ├── valid.p                    # Pickled validation data
│   ├── test.p                     # Pickled test data
│   ├── signnames.csv              # Mapping of class IDs to sign names
├── best_model.h5                  # Saved best model
└── README.md                      # This file
```

---

## 📦 Dataset

I used the **GTSRB** dataset, which contains 50,000+ images of German traffic signs across **43 categories**. For efficiency, I used pickled data files: `train.p`, `valid.p`, and `test.p`.

Download the [data set](https://bitbucket.org/jadslim/german-traffic-signs/src/master/). It contains signnames.csv, training, validation and test set

---

## 🚀 Features Implemented

### ✅ Preprocessing
- Grayscale conversion
- Histogram equalization (to improve contrast)
- Pixel value normalization (0–1 scale)

### ✅ Data Augmentation
Using `ImageDataGenerator`:
- Random horizontal/vertical shifts
- Zoom and shear transformations
- Random rotations

### ✅ CNN Architecture
Custom CNN with:
- 4 Convolution layers
- Batch Normalization for stable learning
- Max Pooling to downsample
- Dropout layers to reduce overfitting
- Fully connected dense layer with Softmax output

### ✅ Training Techniques
- Adam optimizer with a learning rate of 0.001
- EarlyStopping to avoid overfitting
- ModelCheckpoint to save the best model

### ✅ Evaluation
- Accuracy and Loss plots
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- Random sample visualizations of predictions (Correct in green, Incorrect in red)

### ✅ Prediction on External Image
You can pass an image URL, and the model will:
- Download it
- Resize and preprocess it
- Predict the traffic sign
- Display the predicted class and label

---

## 📊 Visualization Examples

### 🔹 Random Samples from Training Set

Displays 10 random preprocessed grayscale images with class ID and label.

### 🔹 Prediction Grid on Test Set

Shows a 5x5 grid of predictions. Colors indicate correctness:
- ✅ Green = Correct
- ❌ Red = Incorrect

---

## 🛠 Requirements

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn opencv-python keras tensorflow
```

Optional: To run in Google Colab, ensure you mount Google Drive.

---

## ▶️ How to Run

### On Google Colab

1. Upload the notebook `Traffic_Sign_Classification.ipynb`.
2. Mount your Google Drive.
3. Ensure `train.p`, `valid.p`, `test.p`, and `signnames.csv` are in the correct folder path.
4. Run cells sequentially.

### On Local Jupyter

1. Clone this repo
2. Place your GTSRB `.p` files and `signnames.csv` inside `GTSRB_dataset/`
3. Run the notebook in Jupyter

---

## 📈 Results

| Metric         | Value |
|----------------|-------|
| Test Accuracy  | ~98%  |
| Val Accuracy   | ~98%  |
| Train Accuracy | ~99%  |

> Note: Exact results vary based on training epochs and random seed.

---

## 🔮 Future Improvements

- Use transfer learning with MobileNet or EfficientNet
- Build a Streamlit UI for real-time predictions

---

## 🧠 Credits

- [GTSRB Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)
- Keras, TensorFlow, OpenCV, Matplotlib, Scikit-learn
- Inspired by Udemy Computer Vision & Deep Learning courses

---

## 📜 License

This project is open-source and free to use under the MIT License.

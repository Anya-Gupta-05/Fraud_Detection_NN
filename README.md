# 💳 Fraud Detection using Neural Network

This project is a simple implementation of a **binary classification model** to detect fraudulent transactions using a fully connected **Neural Network**. The dataset is preprocessed using **label encoding**, and the model is built using **TensorFlow/Keras**.

---

## 🚀 What This Project Does

- Loads and preprocesses a fraud detection dataset
- Encodes categorical values using **Label Encoding**
- Builds and trains a **Neural Network (MLP)** for binary classification
- Evaluates the model and reports accuracy
- Aims to classify transactions as **fraudulent (1)** or **non-fraudulent (0)**

---

## 🧠 Model Architecture

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

Loss Function: binary_crossentropy

Optimizer: adam

Metric: accuracy

🎯 Final Model Accuracy: ~95%


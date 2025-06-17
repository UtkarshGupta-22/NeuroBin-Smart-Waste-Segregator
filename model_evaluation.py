import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random

from google.colab import drive
drive.mount('/content/drive')

model_path = "/content/drive/MyDrive/biodegradable_vs_nonbiodegradable.keras"
model = load_model(model_path)

dataset_path = "/root/.cache/kagglehub/datasets/techsash/waste-classification-data/versions/1/DATASET"
bio_test = os.path.join(dataset_path, "TEST", "O")
nonbio_test = os.path.join(dataset_path, "TEST", "R")

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (96, 96))
    img = img.astype("float32") / 255.0
    return img

X_test, y_test = [], []
for img_name in os.listdir(bio_test):
    X_test.append(preprocess_image(os.path.join(bio_test, img_name)))
    y_test.append(0)
for img_name in os.listdir(nonbio_test):
    X_test.append(preprocess_image(os.path.join(nonbio_test, img_name)))
    y_test.append(1)

X_test = np.array(X_test)
y_test = np.array(y_test)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Biodegradable", "Non-Biodegradable"],
            yticklabels=["Biodegradable", "Non-Biodegradable"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

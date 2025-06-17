import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import drive, files

drive.mount('/content/drive')

model_path = "/content/drive/MyDrive/fine_tuned_waste_classifier_v2.keras"
model = tf.keras.models.load_model(model_path)

categories = ['Biodegradable', 'Non-Biodegradable']

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (96, 96))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

uploaded = files.upload()

for image_name in uploaded.keys():
    image_path = f"/content/{image_name}"
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)

    predicted_class = categories[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    img_display = cv2.imread(image_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    plt.imshow(img_display)
    plt.axis('off')
    plt.title(f"{predicted_class} ({confidence:.2f}%)")
    plt.show()

    print(f"Image: {image_name}")
    print(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)\n")

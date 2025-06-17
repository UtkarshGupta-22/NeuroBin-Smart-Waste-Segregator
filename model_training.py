import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from google.colab import drive
drive.mount('/content/drive')

model_path = "/content/drive/MyDrive/biodegradable_vs_nonbiodegradable.keras"
model = load_model(model_path)

dataset_path = "/root/.cache/kagglehub/datasets/sujaykapadnis/cnn-waste-classification-image-dataset/versions/1"
organic_path = os.path.join(dataset_path, "organic/organic")
recyclable_path = os.path.join(dataset_path, "recyclable/recyclable")

def load_image_paths():
    image_paths, labels = [], []
    for img in os.listdir(organic_path):
        image_paths.append(os.path.join(organic_path, img))
        labels.append(0)
    for img in os.listdir(recyclable_path):
        image_paths.append(os.path.join(recyclable_path, img))
        labels.append(1)
    return np.array(image_paths), np.array(labels)

image_paths, labels = load_image_paths()
image_paths_train, image_paths_val, labels_train, labels_val = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

def preprocess_image(img_path, label):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [96, 96])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = tf.data.Dataset.from_tensor_slices((image_paths_train, labels_train))
train_dataset = train_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((image_paths_val, labels_val))
val_dataset = val_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

for layer in model.layers[:int(len(model.layers) * 0.6)]:
    layer.trainable = False
for layer in model.layers[int(len(model.layers) * 0.6):]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_path = "/content/drive/MyDrive/fine_tuned_waste_classifier.keras"
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=callbacks
)

loss, accuracy = model.evaluate(val_dataset)
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Loss: {loss:.4f}")

final_model_path = "/content/drive/MyDrive/fine_tuned_waste_classifier_v2.keras"
model.save(final_model_path)

tflite_model_path = "/content/drive/MyDrive/fine_tuned_waste_classifier.tflite"
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'], " | Data type:", input_details[0]['dtype'])
print("Output shape:", output_details[0]['shape'], " | Data type:", output_details[0]['dtype'])

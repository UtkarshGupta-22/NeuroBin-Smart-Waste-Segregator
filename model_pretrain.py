import os
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory

DATASET_PATH = "/root/.cache/kagglehub/datasets/rayhanzamzamy/non-and-biodegradable-waste-dataset"

train_dirs = [os.path.join(DATASET_PATH, d) for d in os.listdir(DATASET_PATH) if "TRAIN" in d]
test_dir = os.path.join(DATASET_PATH, "TEST")

IMG_SIZE = (96, 96)
BATCH_SIZE = 32

def load_multiple_datasets(train_dirs):
    all_train_datasets = []
    for train_dir in train_dirs:
        try:
            dataset = image_dataset_from_directory(
                train_dir,
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            all_train_datasets.append(dataset)
        except Exception as e:
            print(f"Error loading {train_dir}: {e}")
    if all_train_datasets:
        merged = all_train_datasets[0]
        for ds in all_train_datasets[1:]:
            merged = merged.concatenate(ds)
        return merged
    return None

train_data = load_multiple_datasets(train_dirs)
val_data = image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

normalization_layer = layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

model = Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(96, 96, 3)),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

EPOCHS = 10
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

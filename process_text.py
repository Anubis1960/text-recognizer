import os
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras import datasets, models

tf.config.threading.set_intra_op_parallelism_threads(8)  # Number of threads for parallel operations


def concatenate_datasets(
        train_inputs1: np.ndarray,
        train_inputs2: np.ndarray,
        train_labels1: np.ndarray,
        train_labels2: np.ndarray,
        test_inputs1: np.ndarray,
        test_inputs2: np.ndarray,
        test_labels1: np.ndarray,
        test_labels2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_combined = np.concatenate((train_inputs1, train_inputs2), axis=0)
    train_targets_comb = np.concatenate((train_labels1, train_labels2), axis=0)

    test_combined = np.concatenate((test_inputs1, test_inputs2), axis=0)
    test_targets_comb = np.concatenate((test_labels1, test_labels2), axis=0)

    return train_combined, train_targets_comb, test_combined, test_targets_comb


def build_weights_dict(labels: np.ndarray) -> dict[int, float]:
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return {i: class_weights[i] for i in range(len(class_weights))}


def build_letters_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load A-Z handwritten dataset
    letters = pd.read_csv('./kaggle/input/A_Z Handwritten Data/A_Z Handwritten Data.csv')

    # Prepare the A-Z data
    inputs = letters.drop('0', axis=1).values / 255.0
    targets = letters['0'].values

    # Shift target labels by 10 for A-Z letters
    train_in, test_in, train_tar, test_tar = train_test_split(inputs, targets, test_size=0.3)

    train_in = train_in.reshape((train_in.shape[0], 28, 28, 1))
    test_in = test_in.reshape((test_in.shape[0], 28, 28, 1))

    print(train_in.shape)

    return train_in, test_in, train_tar, test_tar


def build_digits_dataset(dirr: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_imag_digits = []
    train_label_digits = []

    # Load digit images
    for i in range(10):
        dir_path = os.path.join(dirr, str(i))  # Handle path in a cross-platform manner
        if not os.path.exists(dir_path):  # Check if the directory exists
            continue

        # Add labels for each digit
        train_label_digits += [i] * len(os.listdir(dir_path))

        # Process each image
        for j in os.listdir(dir_path):
            img_path = os.path.join(dir_path, j)  # Handle file path
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue  # Skip if the image cannot be read
                img = cv2.resize(img, (28, 28))
                img = np.invert(img)
                img = img / 255.0  # Normalize
                train_imag_digits.append(img)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue  # Skip on error

    train_imag_digits = np.array(train_imag_digits)

    # Split the dataset into training and testing sets
    train_imag_digits, test_img_digits, train_label_digits, test_lab_digits = train_test_split(
        train_imag_digits, train_label_digits, test_size=0.3)

    # Reshape the images to (28, 28, 1)
    train_imag_digits = train_imag_digits.reshape((train_imag_digits.shape[0], 28, 28, 1))
    test_img_digits = test_img_digits.reshape((test_img_digits.shape[0], 28, 28, 1))

    # Load the MNIST dataset
    (train_mnist, train_labels_mnist), (
        test_mnist, test_labels_mnist) = datasets.mnist.load_data()

    # Normalize the MNIST images
    train_mnist = train_mnist / 255.0
    test_mnist = test_mnist / 255.0

    # Reshape MNIST images to (28, 28, 1)
    train_mnist = train_mnist.reshape((train_mnist.shape[0], 28, 28, 1))
    test_mnist = test_mnist.reshape((test_mnist.shape[0], 28, 28, 1))

    # Combine both datasets
    train_imag_digits = np.concatenate((train_imag_digits, train_mnist), axis=0)
    train_label_digits = np.concatenate((train_label_digits, train_labels_mnist), axis=0)

    test_img_digits = np.concatenate((test_img_digits, test_mnist), axis=0)
    test_lab_digits = np.concatenate((test_lab_digits, test_labels_mnist), axis=0)

    # print(train_images_digits.shape)

    return train_imag_digits, test_img_digits, train_label_digits, test_lab_digits


# Load digit data
path = "kaggle/input/digits/"

train_images_digits, test_images_digits, train_labels_digits, test_labels_digits = build_digits_dataset(path)

# Load A-Z dataset
train_inputs, test_inputs, train_targets, test_targets = build_letters_dataset()

train_targets += 10
test_targets += 10

# Combine the datasets
train_inputs_combined, train_targets_combined, test_inputs_combined, test_targets_combined = concatenate_datasets(
    train_images_digits, train_inputs, train_labels_digits, train_targets, test_images_digits, test_inputs,
    test_labels_digits, test_targets)

class_weight_dict = build_weights_dict(train_targets_combined)

# Shuffle the training data
train_shuffle = np.random.permutation(len(train_inputs_combined))
train_inputs_combined = train_inputs_combined[train_shuffle]
train_targets_combined = train_targets_combined[train_shuffle]

# Shuffle the testing data
test_shuffle = np.random.permutation(len(test_inputs_combined))
test_inputs_combined = test_inputs_combined[test_shuffle]
test_targets_combined = test_targets_combined[test_shuffle]

# Define the CNN model with deeper layers
model = models.Sequential([
    Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1), padding='same'),
    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(100, activation='relu', kernel_initializer='he_uniform'),
    Dropout(0.1),
    Dense(64, activation='relu', kernel_initializer='he_uniform'),
    Dropout(0.125),
    BatchNormalization(),
    Dense(36, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    x=train_inputs_combined,
    y=train_targets_combined,
    validation_data=(test_inputs_combined, test_targets_combined),
    epochs=10,
    class_weight=class_weight_dict,
    validation_split=0.1,
    verbose=1,
    batch_size=64
)

# Print model summary
print(model.summary())

# Plot the loss curves
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Plot the accuracy curves
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_inputs_combined, test_targets_combined)
print(f"Test Accuracy: {test_accuracy}")

# Save the trained model
model.save('text-recognizer-128.keras')

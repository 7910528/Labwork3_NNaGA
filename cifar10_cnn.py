import numpy as np
import matplotlib.pyplot as plt

# Dataset loading
from tensorflow.keras.datasets import cifar10

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical, load_img, img_to_array

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

batch_size = 32  # Number of samples processed together
num_epochs = 5  # Number of training iterations (increase for better results)
kernel_size = 3  # Size of convolutional kernels
pool_size = 2  # Size of pooling windows
conv_depth_1 = 32  # Number of filters in first conv layers
conv_depth_2 = 64  # Number of filters in second conv layers
drop_prob_1 = 0.25  # Dropout probability after pooling
drop_prob_2 = 0.5  # Dropout probability after dense layer
hidden_size = 512  # Number of neurons in fully connected layer

print("\n" + "=" * 60)
print("HYPERPARAMETERS")
print("=" * 60)
print(f"Batch size: {batch_size}")
print(f"Epochs: {num_epochs}")
print(f"Kernel size: {kernel_size}x{kernel_size}")
print(f"Pool size: {pool_size}x{pool_size}")
print(f"Conv depth (layer 1): {conv_depth_1}")
print(f"Conv depth (layer 2): {conv_depth_2}")
print(f"Dropout prob 1: {drop_prob_1}")
print(f"Dropout prob 2: {drop_prob_2}")
print(f"Hidden layer size: {hidden_size}")

print("\n" + "=" * 60)
print("LOADING CIFAR-10 DATASET")
print("=" * 60)

# Load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Get dimensions
num_train, height, width, depth = X_train.shape
num_test = X_test.shape[0]
num_classes = np.unique(y_train).shape[0]

print(f"Training samples: {num_train}")
print(f"Test samples: {num_test}")
print(f"Image shape: {height}x{width}x{depth}")
print(f"Number of classes: {num_classes}")

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

classes_ua = ['літак', 'автомобіль', 'птах', 'кіт', 'олень',
              'собака', 'жаба', 'кінь', 'корабель', 'вантажівка']

# Visualize sample images
print("\nVisualizing sample images...")
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Sample Images from CIFAR-10 Dataset', fontsize=16)

for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(X_train[i])
    ax.set_title(f"{class_names[y_train[i][0]]}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('cifar10_samples.png', dpi=150, bbox_inches='tight')
print("Sample images saved as 'cifar10_samples.png'")
plt.show()

# Convert to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize to [0, 1] range
X_train /= 255.0
X_test /= 255.0

print(f"\nData normalized to range [{X_train.min():.2f}, {X_train.max():.2f}]")

# One-hot encode labels
Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)

print(f"Labels shape: {Y_train.shape}")
print(f"Label encoding example: {y_train[0][0]} -> {Y_train[0]}")

print("\n" + "=" * 60)
print("BUILDING CNN MODEL")
print("=" * 60)

# Input layer
inp = Input(shape=(height, width, depth))

# First convolutional block
conv_1 = Conv2D(conv_depth_1, (kernel_size, kernel_size),
                padding='same', activation='relu')(inp)
conv_2 = Conv2D(conv_depth_1, (kernel_size, kernel_size),
                padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)

# Second convolutional block
conv_3 = Conv2D(conv_depth_2, (kernel_size, kernel_size),
                padding='same', activation='relu')(drop_1)
conv_4 = Conv2D(conv_depth_2, (kernel_size, kernel_size),
                padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)

# Fully connected layers
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

# Create model
model = Model(inputs=inp, outputs=out)

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("\nModel Architecture:")
model.summary()

print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    validation_split=0.1)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Model Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("\nTraining history saved as 'training_history.png'")
plt.show()

print("\n" + "=" * 60)
print("EVALUATING MODEL ON TEST SET")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

# Save architecture as JSON
model_json = model.to_json()
with open("cifar10_model.json", "w") as json_file:
    json_file.write(model_json)
print("Model architecture saved to 'cifar10_model.json'")

# Save weights
model.save_weights("cifar10_model.weights.h5")
print("Model weights saved to 'cifar10_model.weights.h5'")

# Also save complete model
model.save("cifar10_complete_model.keras")
print("Complete model saved to 'cifar10_complete_model.keras'")

print("\n" + "=" * 60)
print("TESTING ON RANDOM TEST IMAGES")
print("=" * 60)

# Select 10 random test images
np.random.seed(42)
test_indices = np.random.choice(num_test, 10, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Predictions on Test Images', fontsize=16)

for i, idx in enumerate(test_indices):
    ax = axes[i // 5, i % 5]

    # Get image and true label
    img = X_test[idx]
    true_label = y_test[idx][0]

    # Make prediction
    img_expanded = np.expand_dims(img, axis=0)
    prediction = model.predict(img_expanded, verbose=0)
    predicted_label = np.argmax(prediction)
    confidence = prediction[0][predicted_label]

    # Display
    ax.imshow(X_test[idx] * 255.0 / 255.0)  # Denormalize for display
    color = 'green' if predicted_label == true_label else 'red'
    ax.set_title(f"True: {class_names[true_label]}\n"
                 f"Pred: {class_names[predicted_label]}\n"
                 f"Conf: {confidence:.2f}",
                 color=color, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('test_predictions.png', dpi=150, bbox_inches='tight')
print("\nTest predictions saved as 'test_predictions.png'")
plt.show()

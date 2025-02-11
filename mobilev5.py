import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# Define directories
dataset_path = "E:\\snake bite\\snakebit_model_traning\\dataset"
batch_size = 32
image_size = (224, 224)  # Resize all images to 224x224

# Data Augmentation & Image Loading
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% validation split
)

# Load Training Data
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',  # Binary classification (0 or 1)
    subset='training'
)

# Load Validation Data
val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Load Pre-Trained Model (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model to keep pre-trained features
base_model.trainable = False  

# Build Custom Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Convert features into a single vector
    layers.Dense(128, activation='relu'),  # Fully connected layer
    layers.Dropout(0.5),  # Prevents overfitting
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

# Train the Model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust epochs as needed
    validation_data=val_generator
)

# Unfreeze the last 20 layers for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Compile again with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy', metrics=['accuracy'])

# Train again with fine-tuning
history_fine = model.fit(
    train_generator,
    epochs=5,  # Fine-tune for fewer epochs
    validation_data=val_generator
)

# Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save("snake_bite_classifier.h5")

# def predict_image(image_path, model):
#     # Load and preprocess the image
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (224, 224))
#     img = img / 255.0  # Normalize
#     img = np.expand_dims(img, axis=0)  # Reshape for model input

#     # Make Prediction
#     prediction = model.predict(img)[0][0]
#     if prediction < 0.5:
#         print("Predicted: Poisonous 🐍")
#     else:
#         print("Predicted: Non-Poisonous ✅")

# # Example usage
# predict_image("E:\\snake bite\\snakebit_model_traning\\test.jpg", model)



# # Load the model
# loaded_model = tf.keras.models.load_model("snake_bite_classifier.h5")


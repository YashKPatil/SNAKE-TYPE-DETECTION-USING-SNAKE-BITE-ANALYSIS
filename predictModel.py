import tensorflow as tf
import cv2
import numpy as np


# Load the previously saved model
loaded_model = tf.keras.models.load_model("E:\\snake bite\\snakebit_model_traning\\snake_bite_classifier.h5")


def predict_with_loaded_model(image_path, model):
    # Load the image from the given path
    img = cv2.imread(image_path)
    
    # Check if image is loaded properly
    if img is None:
        print(f"Error: Unable to load image from {image_path}. Please verify the file path.")
        return None
    
    # Resize the image to match the input size expected by the model (e.g., 224x224)
    img = cv2.resize(img, (224, 224))
    
    # Normalize the image pixel values to be between 0 and 1
    img = img / 255.0
    
    # Expand dimensions to create a batch of one (the model expects a batch input)
    img = np.expand_dims(img, axis=0)
    
    # Use the model to make a prediction
    prediction = model.predict(img)
    return prediction


# Path to your new image for prediction
image_path = "E:\\snake bite\\snakebit_model_traning\\testnon.webp"

# Get the prediction from the model
prediction = predict_with_loaded_model(image_path, loaded_model)

# Check the prediction result
if prediction is not None:
    # Since it's a binary classification with a sigmoid activation, the output will be a probability
    if prediction[0][0] < 0.5:
        print("Predicted: Poisonous 🐍")
    else:
        print("Predicted: Non-Poisonous ✅")

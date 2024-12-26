import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

def predict_with_cnn(model_path, image_path, img_size=128):
    # Load the trained model
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Preprocess the input image
    image = load_img(image_path, target_size=(img_size, img_size))
    image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(image_array)
    class_index = np.argmax(predictions, axis=1)[0]  # Get the class with the highest probability
    confidence = np.max(predictions)  # Get the confidence score

    return class_index, confidence

if __name__ == "__main__":
    model_file = "models/cnn_mask_classifier.h5"  # Path to the trained CNN model
    test_image = "../images/maksssksksss321.png"  # Path to the image you want to predict

    # Classes should match the order used during training
    class_names = ['with_mask', 'without_mask', 'mask_wear_incorrect']

    predicted_class, confidence = predict_with_cnn(model_file, test_image)
    print(f"Predicted class: {class_names[predicted_class]} (Confidence: {confidence * 100:.2f}%)")

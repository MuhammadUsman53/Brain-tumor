import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

MODEL_PATH = 'brain_tumor_model.h5'
IMG_SIZE = (224, 224)

def prepare_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def test_single_image(model, filepath, true_label):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    img = prepare_image(filepath)
    prediction = model.predict(img, verbose=0)
    
    # Class 0: No Tumor, Class 1: Tumor (Based on alphanumeric sort of folders 'no', 'yes')
    # But let's verify logic. In train_model.py, we used flow_from_directory.
    # usually 'no' comes before 'yes'.
    class_names = ['No Tumor', 'Tumor']
    
    predicted_class_index = np.argmax(prediction)
    predicted_label = class_names[predicted_class_index]
    confidence = np.max(prediction) * 100
    
    print(f"--- Testing Image: {filepath} ---")
    print(f"True Label: {true_label}")
    print(f"Predicted: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")
    
    if predicted_label == true_label:
        print("[OK] Correct Prediction")
    else:
        print("[FAIL] Incorrect Prediction")
    print("-" * 30)

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Please run train_model.py first.")
        exit()
        
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    
    # Test on generated test images
    test_single_image(model, "test_image_yes.jpg", "Tumor")
    test_single_image(model, "test_image_no.jpg", "No Tumor")

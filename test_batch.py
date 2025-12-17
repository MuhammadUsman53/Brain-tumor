import tensorflow as tf
import numpy as np
import cv2
import os
import random
from tensorflow.keras.models import load_model

MODEL_PATH = 'brain_tumor_model.h5'
IMG_SIZE = (224, 224)

def prepare_image(filepath):
    img = cv2.imread(filepath)
    if img is None: return None
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def test_batch(model, dir_path, label):
    files = os.listdir(dir_path)
    # Pick 5 random
    files = random.sample(files, min(len(files), 5))
    
    print(f"\n--- Testing {label} Class ---")
    correct = 0
    for f in files:
        path = os.path.join(dir_path, f)
        img = prepare_image(path)
        if img is None: continue
        
        pred = model.predict(img, verbose=0)
        conf = np.max(pred)
        idx = np.argmax(pred)
        # 0=No, 1=Yes
        pred_label = "No Tumor" if idx == 0 else "Tumor"
        
        match = (pred_label == label)
        if match: correct += 1
        
        print(f"{f}: {pred_label} ({conf*100:.1f}%) - {'OK' if match else 'FAIL'}")
    
    print(f"Accuracy on sample: {correct}/{len(files)}")

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    test_batch(model, "data/train/yes", "Tumor")
    test_batch(model, "data/train/no", "No Tumor")

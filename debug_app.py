
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

print("Imports successful")

try:
    if os.path.exists('brain_tumor_model.h5'):
        model = tf.keras.models.load_model('brain_tumor_model.h5')
        print("Model loaded successfully")
        
        # Test prediction
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = img / 255.0
        img_reshape = np.expand_dims(img, axis=0)
        prediction = model.predict(img_reshape)
        print("Prediction successful:", prediction)
    else:
        print("Model file not found")

except Exception as e:
    print("Error occurred:")
    print(e)

import os
from sklearn.utils import class_weight
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2
import shutil
import argparse

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "data"
MODEL_PATH = "brain_tumor_model.h5"

# Dummy data generation removed as per user request. 
# Dataset must be provided by the user in 'data/train' and 'data/val'.

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import SGD

def build_model():
    # Use MobileNetV2 - Much smaller and faster for Cloud Deployment
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Freeze the base model initially
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found.")
        return

    # Use MobileNetV2 specific preprocessing (scales to [-1, 1])
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30, 
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Ensure you have 'train' and 'val' directories inside 'data'.")
        return

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    model = build_model()
    
    # Phase 1: Train Head
    print("--- Phase 1: Training Head (Frozen Base) ---")
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights_vals))
    print(f"Computed Class Weights: {class_weights}")

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > BATCH_SIZE else 1,
        epochs=EPOCHS, 
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > BATCH_SIZE else 1,
        class_weight=class_weights
    )
    
    # Phase 2: Fine-Tuning
    print("\n--- Phase 2: Fine-Tuning (Unfreezing Model) ---")
    
    # Unfreeze the base model for fine-tuning
    # Since we used functional API input=base_model.input, we iterate model.layers
    
    # Set all layers to trainable = True first, then freeze early ones
    model.trainable = True
    
    # Freeze all layers except the last 40 layers of the entire model
    # (MobileNetV2 has > 150 layers)
    for layer in model.layers[:-40]:
        layer.trainable = False
            
    # Recompile with lower LR
    model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > BATCH_SIZE else 1,
        epochs=EPOCHS, 
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > BATCH_SIZE else 1,
        class_weight=class_weights
    )
    
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()


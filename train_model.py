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
EPOCHS = 10
DATA_DIR = "data"
MODEL_PATH = "brain_tumor_model.h5"

def create_dummy_data():
    """Creates a dummy dataset with MRI-like shapes (Ellipse for brain)."""
    print("Creating dummy dataset for demonstration...")
    categories = ["yes", "no"]
    for category in categories:
        path = os.path.join(DATA_DIR, "train", category)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        
        # Create 100 dummy images per category
        for i in range(100):
            # 1. Background: Black
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # 2. "Brain": Gray Ellipse
            center = (112, 112)
            axes = (80, 100) # Width, Height
            angle = 0
            color = (100, 100, 100) # Gray
            cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
            
            # Add some variability/noise
            noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

            # 3. Tumor: Bright Circle (Only for 'yes')
            if category == "yes":
                # Random position within the brain area
                t_x = np.random.randint(80, 144)
                t_y = np.random.randint(60, 164)
                cv2.circle(img, (t_x, t_y), 15, (200, 200, 200), -1)
            
            cv2.imwrite(os.path.join(path, f"dummy_{i}.jpg"), img)
            
    # Copy some to val
    for category in categories:
        val_path = os.path.join(DATA_DIR, "val", category)
        if os.path.exists(val_path):
            shutil.rmtree(val_path)
        train_path = os.path.join(DATA_DIR, "train", category)
        os.makedirs(val_path, exist_ok=True)
        # Copy 20 images
        for i in range(20):
            shutil.copy(os.path.join(train_path, f"dummy_{i}.jpg"), 
                        os.path.join(val_path, f"dummy_{i}.jpg"))
    
    # Generate Test Images for the User
    # Tumor Image
    img_yes = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.ellipse(img_yes, (112, 112), (80, 100), 0, 0, 360, (100, 100, 100), -1)
    cv2.circle(img_yes, (112, 80), 15, (200, 200, 200), -1)
    cv2.imwrite("test_image_yes.jpg", img_yes)
    
    # No Tumor Image
    img_no = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.ellipse(img_no, (112, 112), (80, 100), 0, 0, 360, (100, 100, 100), -1)
    cv2.imwrite("test_image_no.jpg", img_no)

    print("Dummy dataset and test images (test_image_yes.jpg, test_image_no.jpg) created.")

from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD

def build_model():
    # Use VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Freeze the base model initially
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found.")
        return

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
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
    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9),
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
        epochs=10, 
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > BATCH_SIZE else 1,
        class_weight=class_weights
    )
    
    # Phase 2: Fine-Tuning
    print("\n--- Phase 2: Fine-Tuning (Unfreezing Block 5) ---")
    
    # Find the VGG16 base layer (it's the input to the model, or we can access via layers)
    # Since we used Functional API with `base_model.input`, the layers are in `model.layers`.
    # VGG16 layers are flattened in `model.layers`? 
    # Let's verify. VGG16(include_top=False) returns a Model. 
    # build_model() calls `base_model.output`.
    # In Functional API, `base_model`'s layers are part of the graph.
    # To set them trainable, we need to iterate over the layers of the base_model we used.
    # But `base_model` is local.
    # We can fetch it by name or index. The first layer is Input, then blocks.
    # Actually, Keras usually nests the model if checking `model.layers`? 
    # No, `base_model.output` connects the graph.
    # Let's iterate `model.layers`.
    
    # We want to unfreeze block5 (conv layers).
    # VGG16 has 5 blocks. block5_conv1, block5_conv2, block5_conv3.
    
    for layer in model.layers:
        if 'block5' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False # Keep others frozen
            
    # Recompile with lower LR
    model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > BATCH_SIZE else 1,
        epochs=10, 
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > BATCH_SIZE else 1,
        class_weight=class_weights
    )
    
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Create dummy data for demonstration")
    args = parser.parse_args()
    
    if args.demo:
        create_dummy_data()
        
    train()


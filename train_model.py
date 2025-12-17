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

def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Freeze the base model
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x) # 2 classes: yes or no
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found.")
        print("Please create a 'data' folder with 'train' and 'val' subfolders,")
        print("containing 'yes' (tumor) and 'no' (no tumor) image folders.")
        print("Or run with --demo to generate dummy data.")
        return

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
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
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Calculate class weights

    
    # Get class indices to verify 'no' is 0 and 'yes' is 1
    print("Class indices:", train_generator.class_indices)
    
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights_vals))
    print(f"Computed Class Weights: {class_weights}")

    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > BATCH_SIZE else 1,
        epochs=EPOCHS + 10, # Increase epochs from 10 to 20
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > BATCH_SIZE else 1,
        class_weight=class_weights
    )
    
    # --- FINE TUNING ---
    print("\nStarting Fine-Tuning...")
    # This `model` is the Functional model we built: inputs -> base_model -> ...
    # model.layers[1] is typically base_model (MobileNetV2) because input is layer 0
    # Let's find the MobileNetV2 layer by name or index.
    # In build_model(): base_model = MobileNetV2(...) which is passed as input? No.
    # inputs=base_model.input
    # The Model object 'model' wraps the graph.
    # The layers of 'model' essentially flatten the graph unless it's a subclassed model with nested layers.
    # BUT, MobileNetV2 is a Functional model itself. When we do x = base_model.output, we are hooking into it.
    
    # Actually, in Keras Functional API, if we use `x = base_model.output`, the base_model layers are part of the graph.
    # However, to freeze specific layers OF the base_model, we should iterate over `base_model.layers`.
    # We need to get the `base_model` object instance again or find it.
    # In `build_model`, we created `base_model` but didn't attach it as a property.
    # But `model.layers` should contain the layers.
    # WAIT. `model.layers` in the resulting model might include the *entire* MobileNet as a single layer if we used `base_model(input)`?
    # NO, we used `x = base_model.output`. This means the individual layers of MobileNet are NOT in `model.layers`? 
    # Actually, usually they are NOT if we just used the output tensor.
    # However, `base_model` variable is local to `build_model`. 
    # We need to access the layers that *make up* the model.
    
    # If we constructed the model as `Model(inputs=base_model.input, ...)`:
    # The `model` effectively *is* the extended MobileNet.
    # So we can just iterate `model.layers`.
    
    # Let's check the layer names.
    # MobileNet layers usually start with 'input_', 'Conv1', 'expanded_conv_', etc.
    # We want to freeze the early ones.
    
    # Let's try iterating model.layers directly.
    fine_tune_at = 100
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # But wait, we set `base_model.trainable = True`... where `base_model` reference is lost.
    # We need to set all layers to trainable = True first, then freeze early ones.
    model.trainable = True
        
    # Recompile with very low learning rate
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=1e-5),
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


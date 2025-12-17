import os
# Force CPU usage to prevent GPU init crashes
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageOps
import numpy as np
import cv2
import traceback
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .tumor {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .no-tumor {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def build_model():
    # Rebuild the exact same architecture as training
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False # Initial state matches
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def load_model():
    try:
        # Load weights only to bypass architecture version mismatches
        model = build_model()
        model.load_weights('brain_tumor_model.h5')
        return model, None
    except Exception as e:
        return None, str(e)

def import_and_predict(image_data, model):
    size = (224, 224)    
    image = image_data.resize(size)
    # image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS) 
    img = np.asarray(image)
    # img = img / 255.0  <-- Remove old scaling
    
    # Preprocess using MobileNetV2 logic (expects 0-255 inputs and scales internally)
    img = preprocess_input(img)
    
    img_reshape = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img_reshape)
    return prediction

# Sidebar for additional info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964063.png", width=100)
    st.title("Brain Tumor Detector")
    st.info("This application uses a deep learning model to detect brain tumors from MRI scans.")
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("1. Upload a brain MRI image (JPG, PNG, JPEG).")
    st.markdown("2. The system will analyze the image.")
    st.markdown("3. View the prediction result.")
    st.markdown("---")
    st.markdown("Created by Muhammad Usman")

# Main Page Content
st.markdown("<h1>Brain Tumor Detection System (AI Powered) v2.0</h1>", unsafe_allow_html=True)

# Check if model file exists but is invalid
model, load_err = load_model()

if model is None:
    # If file exists but load failed, it's corrupt
    if os.path.exists('brain_tumor_model.h5'):
        st.warning(f"⚠️ Model found but failed to load: {load_err}. Regenerating...")
        try:
            os.remove('brain_tumor_model.h5')
        except PermissionError:
            st.error("⚠️ Cannot remove corrupt model file. It is currently in use. Please delete 'brain_tumor_model.h5' manually and refresh.")
            st.stop()
        except FileNotFoundError:
            pass
        except Exception as e:
            st.warning(f"Could not remove model file: {e}")
        st.cache_resource.clear()
    
    
    # Check for model existence
    if not os.path.exists('brain_tumor_model.h5'):
         st.error("❌ Model not found. Please ensure 'brain_tumor_model.h5' exists or train the model using your own dataset.")
         st.info("To train: Place your dataset in the 'data' folder and run 'python train_model.py'.")
         st.stop()
    
    # Generate if missing (Logic Removed per user request)
    # if not os.path.exists('brain_tumor_model.h5'): ...

# Retry load
model, load_err_2 = load_model()

if model is None:
    st.error(f"❌ Critical Error: Model failed to load.")
    st.error(f"Details: {load_err_2}")
    st.info("If you are on Streamlit Cloud, please ensure 'brain_tumor_model.h5' is uploaded or 'data' folder is present.")
    st.stop()

file = st.file_uploader("Upload an MRI Scan", type=["jpg", "png", "jpeg"])

if file is None:
    st.markdown(
        """
        <div style="text-align: center; color: #7f8c8d; padding: 50px;">
            <p>Please upload an image to get started.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    try:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True, caption="Uploaded MRI Scan")
    except Exception as e:
        st.error(f"Error opening image. Please ensure it is a valid JPG/PNG file. Details: {e}")
        st.stop()
    
    detect_btn = st.button("Detect Tumor")
    
    if detect_btn:
        with st.spinner("Analyzing image..."):
            try:
                predictions = import_and_predict(image, model)
                class_names = ['No Tumor', 'Tumor']
                
                # Depending on how the generator loads classes, 'no' usually comes before 'yes' alphabetically.
                # In train_model.py: categories are potentially sorted. 
                # ImageDataGenerator uses sorted alphanumeric order.
                # So if folders are 'no' and 'yes':
                # 0 -> no
                # 1 -> yes (Tumor)
                
                # Let's double check this logic in training effectively, but standard is alphanumeric.
                
                predicted_class = class_names[np.argmax(predictions)]
                confidence = np.max(predictions) * 100
                
                if predicted_class == 'Tumor':
                    st.markdown(
                        f"""
                        <div class="prediction-box tumor">
                            ⚠️ Prediction: Tumor<br>
                            <span style="font-size: 20px; font-weight: normal;">Confidence: <strong>{confidence:.1f}%</strong></span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="prediction-box no-tumor">
                            ✅ Prediction: No Tumor<br>
                            <span style="font-size: 20px; font-weight: normal;">Confidence: <strong>{confidence:.1f}%</strong></span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Dynamic Progress Bar Color
                st.write("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model Confidence", f"{confidence:.1f}%")
                with col2:
                    if confidence > 80:
                         st.success("High Confidence Analysis")
                    else:
                         st.warning("Low Confidence Analysis")
                
                st.progress(int(confidence))
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.write("Make sure the image is a valid RGB image.")



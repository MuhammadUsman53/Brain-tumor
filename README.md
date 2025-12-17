# 🧠 Brain Tumor Detection System

This project is a Deep Learning application that detects brain tumors from MRI scans using a Convolutional Neural Network (CNN). The application is built using TensorFlow/Keras and deployed using Streamlit.

## 📁 Project Structure

- `app.py`: The main Streamlit application for the user interface.
- `train_model.py`: Script to train the CNN model (MobileNetV2).
- `requirements.txt`: List of Python dependencies.
- `brain_tumor_model.h5`: The trained model file (generated after training).
- `data/`: Directory where the dataset should be stored.

## 🚀 Setup & Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Dataset:**
    - You need a Brain Tumor MRI dataset.
    - Recommended: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) or any similar dataset with "Tumor" and "No Tumor" classes.
    - Extract the dataset into a folder named `data`.
    - The structure should look like this:
        ```
        data/
        ├── train/
        │   ├── no/
        │   └── yes/
        └── val/
            ├── no/
            └── yes/
        ```
    - *Note: If your dataset has different folder names, please rename them to `no` (no tumor) and `yes` (tumor) inside `train` and `val` directories.*

    **Testing with Dummy Data:**
    If you don't have a dataset yet, you can generate a dummy dataset for testing the pipeline:
    ```bash
    python train_model.py --demo
    ```

3.  **Train the Model:**
    Run the training script to build and save the model:
    ```bash
    python train_model.py
    ```
    This will save the model as `brain_tumor_model.h5`.

4.  **Run the Application:**
    Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## 🛠️ Technologies Used
- **Python**
- **TensorFlow / Keras** (Deep Learning)
- **MobileNetV2** (Transfer Learning)
- **Streamlit** (Web Interface)
- **OpenCV & NumPy** (Image Processing)

## 📝 Assignment Detail
This project fulfills the requirements for the Python + Deep Learning Assignment:
- Uses a pre-trained CNN (MobileNetV2).
- Validates on Tumor/No Tumor classes.
- Includes a user-friendly Streamlit interface.

## 🌐 Deployment to Streamlit Cloud

1.  **Push to GitHub:**
    - Initialize a git repository: `git init`
    - Add files: `git add .`
    - Commit: `git commit -m "Initial commit"`
    - Push to a new public repository on GitHub.

2.  **Deploy:**
    - Go to [Streamlit Cloud](https://streamlit.io/cloud).
    - Sign in with GitHub.
    - Click "New app".
    - Select your repository, branch, and `app.py` as the main file.
    - Click "Deploy".

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import nibabel as nib
import io
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nibabel import FileHolder

# Load the trained model
model = tf.keras.models.load_model('stroke_detection_cnn_model.h5')

# Preprocessing function for uploaded images
def preprocess_image(image_data, target_size=(128, 128)):
    """Normalize and resize the image data."""
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    resized_image = tf.image.resize(image_data, target_size)
    return resized_image.numpy()

# Function to calculate the Dice coefficient
def dice_coefficient(y_true, y_pred, threshold=0.5):
    """Calculate the Dice coefficient (similar to F1-score for image segmentation)."""
    y_pred_binary = (y_pred > threshold).astype(int)
    y_true_binary = (y_true > 0).astype(int)

    intersection = np.sum(y_true_binary * y_pred_binary)
    dice = (2. * intersection) / (np.sum(y_true_binary) + np.sum(y_pred_binary))
    return dice

# Function to compile multiple metrics
def compile_metrics(y_true, y_pred, threshold=0.5):
    """Compile multiple metrics including accuracy, precision, recall, F1-score, and Dice coefficient."""
    
    # Binarize predictions and true masks based on the threshold
    y_pred_binary = (y_pred > threshold).astype(int)
    y_true_binary = (y_true > 0).astype(int)

    # Flatten arrays to work with sklearn metrics
    y_true_flat = y_true_binary.flatten()
    y_pred_flat = y_pred_binary.flatten()

    # Calculate metrics
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=1)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=1)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=1)
    dice = dice_coefficient(y_true, y_pred, threshold)

    # Return metrics as a dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "dice_coefficient": dice
    }

# Function to predict and visualize the result, along with evaluation metrics
def predict_and_visualize(model, image, true_mask, image_name):
    """Preprocess the image, predict the mask, evaluate, and visualize the result."""
    # Preprocess the image
    image_preprocessed = preprocess_image(image)
    image_preprocessed = np.expand_dims(image_preprocessed, axis=0)  # Add batch dimension

    # Perform prediction
    prediction = model.predict(image_preprocessed)
    predicted_mask = prediction[0]

    # Calculate metrics
    metrics = compile_metrics(true_mask, predicted_mask)

    # Plot original image, true mask, and predicted mask
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title(f"Original Image - {image_name}")
    ax[1].imshow(true_mask, cmap='gray')
    ax[1].set_title(f"True Mask - {image_name}")
    ax[2].imshow(predicted_mask, cmap='gray')
    ax[2].set_title(f"Predicted Mask - {image_name}")
    st.pyplot(fig)

    # Display metrics
    st.write(f"Metrics for {image_name}:")
    st.write(f"**Accuracy**: {metrics['accuracy']:.4f}")
    st.write(f"**Precision**: {metrics['precision']:.4f}")
    st.write(f"**Recall (Sensitivity)**: {metrics['recall']:.4f}")
    st.write(f"**F1-Score**: {metrics['f1_score']:.4f}")
    st.write(f"**Dice Coefficient**: {metrics['dice_coefficient']:.4f}")

# Streamlit UI elements
st.title("Stroke Segmentation Model with Evaluation Metrics")
st.write("Upload multiple validating images and ground truth masks to test the model and evaluate its performance.")

# File uploader for multiple images and true masks
uploaded_images = st.file_uploader("Choose image files", type=["nii", "gz"], accept_multiple_files=True)
uploaded_masks = st.file_uploader("Choose true mask files", type=["nii", "gz"], accept_multiple_files=True)

# Only show the "Run" button if images and masks are uploaded and the count matches
if uploaded_images and uploaded_masks and len(uploaded_images) == len(uploaded_masks):
    # Add a button to trigger the model prediction and display the results
    if st.button('Run Model for Prediction'):
        # Iterate through the uploaded images and masks
        for uploaded_image, uploaded_mask in zip(uploaded_images, uploaded_masks):
            try:
                # Use BytesIO to handle the uploaded files
                image_bytes = io.BytesIO(uploaded_image.read())
                mask_bytes = io.BytesIO(uploaded_mask.read())

                # Load the NIfTI image and true mask using nibabel's FileHolder
                image_file_holder = FileHolder(fileobj=image_bytes)
                mask_file_holder = FileHolder(fileobj=mask_bytes)

                # Load the NIfTI images from memory
                nifti_image = nib.Nifti1Image.from_file_map({'header': image_file_holder, 'image': image_file_holder})
                img_data = nifti_image.get_fdata()

                nifti_mask = nib.Nifti1Image.from_file_map({'header': mask_file_holder, 'image': mask_file_holder})
                true_mask_data = nifti_mask.get_fdata()

                image_name = uploaded_image.name
                st.write(f"Image {image_name} uploaded successfully")

                # Predict and visualize for each image along with metrics
                st.write(f"Prediction and Evaluation Results for {image_name}:")
                predict_and_visualize(model, img_data, true_mask_data, image_name)

            except nib.filebasedimages.ImageFileError as e:
                st.error(f"Failed to load image: {uploaded_image.name}. Please ensure the file is a valid NIfTI file.")
            except Exception as e:
                st.error(f"An error occurred while processing {uploaded_image.name}: {str(e)}")

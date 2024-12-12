# AI-Project
1.
Referencing:
https://github.com/npnl/ATLAS/
https://www.nature.com/articles/s41597-022-01923-0


3. Data Preparation
MRI images from ATLAS referenced above contains about 300 images with thier corresponding masks in training folder. The images were trained using U-NET model.

Input: 3D .nii.gz images for T1-weighted MRIs and lesion masks.
Output: Preprocessed 2D slices resized to (128, 128).
Steps:

Load .nii.gz images using nibabel.
Extract the middle slice along the z-axis.
Normalize intensity values to [0, 1].
Resize images to (128, 128) using TensorFlow.
3. Data Augmentation

Augment the training data to improve generalization:
Random rotations, flips, shifts, and zooms.
Generate 3 augmented samples per original image.
4. U-Net Model Design

Implement U-Net architecture with:
Encoder: Convolutions + MaxPooling.
Bottleneck: High-level feature extraction with Dropout.
Decoder: Transposed convolutions for upsampling.
Output: Sigmoid activation for binary segmentation.
Loss Function:
Combined Loss = Binary Cross-Entropy + Dice Loss.
5. Training

Split data into training and validation sets.
Use augmented data for training.
Train the model with:
EarlyStopping to monitor validation loss and prevent overfitting.
20 epochs (adjustable).
6. Evaluation

Predict lesion masks for the validation set.
Threshold predictions at 0.5 for binary masks.
Compute the following metrics:
Accuracy
Precision
Recall
Dice Coefficient
Jaccard Index
Generate a confusion matrix to evaluate performance visually.
7. Visualization

Display:
Input MRI slices.
Ground truth masks.
Predicted masks.
Overlay predictions on input images for better insight.

# BlurBuddy - Adaptive Face-Blur for Privacy

A machine learning project that implements face identification in videos for adaptive face-blur privacy systems. This is the first part of a larger system that will eventually include face detection/landmarks, optical flow for temporal tracking, and style-transfer/face-to-avatar morphing.

## Project Overview

The goal is to replace faces in videos with consistent pseudonymous avatars per person without revealing identity, while potentially adding lip-sync while removing identity.

## Current Implementation

This implementation focuses on building a neural network model that can identify faces in videos using:

1. **Face Images**: RGB face images from video frames
2. **Facial Landmarks**: 68-point facial landmarks for geometric features
3. **Combined Features**: Fusion of both image and landmark features

## Model Architecture

The `FaceIdentificationModel` combines:

- **CNN Branch**: Processes face images through convolutional layers
- **MLP Branch**: Processes facial landmarks through fully connected layers
- **Fusion Layer**: Combines both feature types for final classification

## Files Structure

```
BlurBuddy/
├── Blur_Buddy.ipynb           # Main notebook for running the model
├── face_identification_model.py   # Main model implementation and training code
├── test_model.py             # Test script for the model
├── Data/                     # Dataset directory
│   ├── youtube_faces_with_keypoints_full_1/
│   ├── youtube_faces_with_keypoints_full_2/
│   ├── youtube_faces_with_keypoints_full_3/
│   ├── youtube_faces_with_keypoints_full_4/
│   └── youtube_faces_with_keypoints_full.csv
└── README.md                 # This file
```

## Dataset

The project uses the YouTube Faces dataset with keypoints, stored as `.npz` files containing:

- `colorImages`: RGB face images (H, W, 3, T) where T is number of frames
- `landmarks2D`: 2D facial landmarks (68, 2, T)
- `landmarks3D`: 3D facial landmarks (68, 3, T)
- `boundingBox`: Face bounding box coordinates

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision torchaudio pandas scikit-learn matplotlib pillow opencv-python numpy
   ```

2. **Run the Main Notebook**:
   - Open `Blur_Buddy.ipynb` in a Jupyter notebook environment
   - Execute the cells to start training

3. **Or, Run the Training Script Directly**:
   ```bash
   python face_identification_model.py
   ```

4. **Test the Model**:
   ```bash
   python test_model.py
   ```

## Training Process

The training pipeline:
1. Loads face data from NPZ files
2. Preprocesses images and landmarks
3. Splits data into train/validation/test sets
4. Trains the model with early stopping and learning rate scheduling
5. Evaluates on test set and generates classification report

## Model Features

- **Multi-modal Input**: Combines visual and geometric features
- **Robust Training**: Uses early stopping and learning rate scheduling
- **Comprehensive Evaluation**: Includes accuracy metrics and classification reports
- **Easy Inference**: Provides `identify_face()` function for video frame identification

## Future Extensions

This is the foundation for the complete video privacy system. Future work will include:

1. **Face Detection**: Automatic face detection in video frames
2. **Temporal Tracking**: Optical flow for consistent tracking across frames
3. **Avatar Generation**: Style-transfer or face-to-avatar morphing
4. **Lip-sync Preservation**: Maintaining lip movements while hiding identity

## Contributing

Feel free to contribute by:
- Improving model architecture
- Adding new features
- Fixing bugs
- Enhancing documentation

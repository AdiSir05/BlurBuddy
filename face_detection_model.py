"""
Face Detection Model for Video Privacy System

This module implements a machine learning model for detecting faces in videos
as part of an adaptive face-blur privacy system.

Overview:
- Loads face data from YouTube dataset (images, bounding boxes, landmarks)
- Creates a neural network model to detect faces and predict bounding boxes + landmarks
- Trains the model to detect faces consistently across video frames
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class FaceDetectionDataset(Dataset):
    """Dataset class for loading face images and ground truth bounding boxes/landmarks."""
    
    def __init__(self, df, data_dir, transform=None, target_size=(224, 224)):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get row data
        row = self.df.iloc[idx]
        npz_file = row['file_path']
        
        try:
            data = np.load(npz_file)
            
            # Get a random frame from the video
            num_frames = data['colorImages'].shape[3]
            frame_idx = np.random.randint(0, num_frames)
            
            # Get face image
            face_image = data['colorImages'][:, :, :, frame_idx]  # Shape: (H, W, 3)
            original_h, original_w = face_image.shape[:2]
            
            # Get ground truth bounding box and landmarks for this frame
            bbox = data['boundingBox'][:, :, frame_idx]  # Shape: (4, 2) - 4 corner points
            landmarks = data['landmarks2D'][:, :, frame_idx]  # Shape: (68, 2)
            
            # Normalize bounding box to [0, 1] range
            bbox_normalized = np.zeros(4)  # [x_min, y_min, x_max, y_max]
            bbox_normalized[0] = bbox[:, 0].min() / original_w  # x_min
            bbox_normalized[1] = bbox[:, 1].min() / original_h  # y_min
            bbox_normalized[2] = bbox[:, 0].max() / original_w  # x_max
            bbox_normalized[3] = bbox[:, 1].max() / original_h  # y_max
            
            # Normalize landmarks to [0, 1] range
            landmarks_normalized = landmarks.copy()
            landmarks_normalized[:, 0] /= original_w  # x coordinates
            landmarks_normalized[:, 1] /= original_h  # y coordinates
            
            # Resize image to target size
            face_image_resized = cv2.resize(face_image, self.target_size)
            
            # Convert image to tensor and normalize
            face_image_tensor = torch.from_numpy(face_image_resized.astype(np.float32)).permute(2, 0, 1) / 255.0
            
            # Convert ground truth to tensors
            bbox_tensor = torch.from_numpy(bbox_normalized.astype(np.float32))
            landmarks_tensor = torch.from_numpy(landmarks_normalized.astype(np.float32)).flatten()  # Flatten to 136 features
            
            # Apply transform if provided
            if self.transform:
                face_image_tensor = self.transform(face_image_tensor)
            
            return face_image_tensor, bbox_tensor, landmarks_tensor
            
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            # Return dummy data in case of error
            dummy_image = torch.zeros((3, self.target_size[0], self.target_size[1]))
            dummy_bbox = torch.zeros(4)
            dummy_landmarks = torch.zeros(136)
            return dummy_image, dummy_bbox, dummy_landmarks


class FaceDetectionModel(nn.Module):
    """CNN model for face detection (bounding box + landmarks prediction)."""
    
    def __init__(self, num_landmarks=68):
        super(FaceDetectionModel, self).__init__()
        
        # Backbone CNN for feature extraction
        self.backbone = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224x224 -> 112x112
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112 -> 56x56
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            
            # Fifth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),  # 14x14 -> 7x7
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 4),  # [x_min, y_min, x_max, y_max]
            nn.Sigmoid()  # Ensure outputs are in [0, 1] range
        )
        
        # Landmarks regression head
        self.landmarks_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_landmarks * 2),  # 68 landmarks * 2 coordinates
            nn.Sigmoid()  # Ensure outputs are in [0, 1] range
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Global pooling
        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Predict bounding box
        bbox_pred = self.bbox_head(pooled_features)
        
        # Predict landmarks
        landmarks_pred = self.landmarks_head(pooled_features)
        
        return bbox_pred, landmarks_pred


def train_epoch(model, dataloader, criterion_bbox, criterion_landmarks, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    bbox_losses = []
    landmarks_losses = []
    
    for batch_idx, (images, bbox_gt, landmarks_gt) in enumerate(dataloader):
        images = images.to(device)
        bbox_gt = bbox_gt.to(device)
        landmarks_gt = landmarks_gt.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        bbox_pred, landmarks_pred = model(images)
        
        # Calculate losses
        bbox_loss = criterion_bbox(bbox_pred, bbox_gt)
        landmarks_loss = criterion_landmarks(landmarks_pred, landmarks_gt)
        
        # Combined loss (you can adjust these weights)
        total_loss_batch = bbox_loss + 0.5 * landmarks_loss
        
        # Backward pass
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        bbox_losses.append(bbox_loss.item())
        landmarks_losses.append(landmarks_loss.item())
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, '
                  f'Total Loss: {total_loss_batch.item():.4f}, '
                  f'Bbox Loss: {bbox_loss.item():.4f}, '
                  f'Landmarks Loss: {landmarks_loss.item():.4f}')
    
    avg_total_loss = total_loss / len(dataloader)
    avg_bbox_loss = np.mean(bbox_losses)
    avg_landmarks_loss = np.mean(landmarks_losses)
    
    return avg_total_loss, avg_bbox_loss, avg_landmarks_loss


def evaluate(model, dataloader, criterion_bbox, criterion_landmarks, device):
    """Evaluate the model on validation/test set."""
    model.eval()
    total_loss = 0.0
    bbox_losses = []
    landmarks_losses = []
    
    with torch.no_grad():
        for images, bbox_gt, landmarks_gt in dataloader:
            images = images.to(device)
            bbox_gt = bbox_gt.to(device)
            landmarks_gt = landmarks_gt.to(device)
            
            # Forward pass
            bbox_pred, landmarks_pred = model(images)
            
            # Calculate losses
            bbox_loss = criterion_bbox(bbox_pred, bbox_gt)
            landmarks_loss = criterion_landmarks(landmarks_pred, landmarks_gt)
            
            # Combined loss
            total_loss_batch = bbox_loss + 0.5 * landmarks_loss
            
            total_loss += total_loss_batch.item()
            bbox_losses.append(bbox_loss.item())
            landmarks_losses.append(landmarks_loss.item())
    
    avg_total_loss = total_loss / len(dataloader)
    avg_bbox_loss = np.mean(bbox_losses)
    avg_landmarks_loss = np.mean(landmarks_losses)
    
    return avg_total_loss, avg_bbox_loss, avg_landmarks_loss


def visualize_predictions(model, dataloader, device, num_samples=4):
    """Visualize model predictions on sample images."""
    model.eval()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    with torch.no_grad():
        for i, (images, bbox_gt, landmarks_gt) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            bbox_gt = bbox_gt.to(device)
            landmarks_gt = landmarks_gt.to(device)
            
            # Get predictions
            bbox_pred, landmarks_pred = model(images)
            
            # Convert to numpy for visualization
            image = images[0].cpu().permute(1, 2, 0).numpy()
            bbox_gt_np = bbox_gt[0].cpu().numpy()
            landmarks_gt_np = landmarks_gt[0].cpu().numpy().reshape(68, 2)
            bbox_pred_np = bbox_pred[0].cpu().numpy()
            landmarks_pred_np = landmarks_pred[0].cpu().numpy().reshape(68, 2)
            
            # Denormalize coordinates
            h, w = image.shape[:2]
            bbox_gt_np *= [w, h, w, h]
            bbox_pred_np *= [w, h, w, h]
            landmarks_gt_np *= [w, h]
            landmarks_pred_np *= [w, h]
            
            # Plot ground truth
            axes[0, i].imshow(image)
            axes[0, i].set_title(f'Sample {i+1} - Ground Truth')
            
            # Draw ground truth bounding box
            gt_rect = patches.Rectangle(
                (bbox_gt_np[0], bbox_gt_np[1]), 
                bbox_gt_np[2] - bbox_gt_np[0], 
                bbox_gt_np[3] - bbox_gt_np[1],
                linewidth=2, edgecolor='green', facecolor='none'
            )
            axes[0, i].add_patch(gt_rect)
            
            # Draw ground truth landmarks
            axes[0, i].scatter(landmarks_gt_np[:, 0], landmarks_gt_np[:, 1], 
                             c='green', s=2, alpha=0.7)
            
            # Plot predictions
            axes[1, i].imshow(image)
            axes[1, i].set_title(f'Sample {i+1} - Predictions')
            
            # Draw predicted bounding box
            pred_rect = patches.Rectangle(
                (bbox_pred_np[0], bbox_pred_np[1]), 
                bbox_pred_np[2] - bbox_pred_np[0], 
                bbox_pred_np[3] - bbox_pred_np[1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[1, i].add_patch(pred_rect)
            
            # Draw predicted landmarks
            axes[1, i].scatter(landmarks_pred_np[:, 0], landmarks_pred_np[:, 1], 
                             c='red', s=2, alpha=0.7)
            
            axes[0, i].axis('off')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def detect_face(model, image_tensor, device):
    """Detect face in a single image tensor."""
    model.eval()
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        bbox_pred, landmarks_pred = model(image_batch)
        
        bbox = bbox_pred[0].cpu().numpy()
        landmarks = landmarks_pred[0].cpu().numpy().reshape(68, 2)
        
        return bbox, landmarks


print("Face Detection Model loaded successfully!")
print(f"Using device: {device}")

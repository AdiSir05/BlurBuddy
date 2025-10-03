"""
Face Identification Model for Video Privacy System

This module implements a machine learning model for identifying faces in videos
as part of an adaptive face-blur privacy system.

Overview:
- Loads face data from YouTube dataset (images and landmarks)
- Creates a neural network model combining image and landmark features
- Trains the model to identify faces consistently across video frames
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class FaceDataset(Dataset):
    """Dataset class for loading face images and landmarks from npz files."""
    
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        
        # Encode person names to numerical labels
        self.label_encoder = LabelEncoder()
        # Handle both 'personName' and 'person_name' column names
        person_col = 'personName' if 'personName' in df.columns else 'person_name'
        self.df['encoded_label'] = self.label_encoder.fit_transform(df[person_col])
        
        # Split npz files by folder
        self.folder_map = {
            1: 'youtube_faces_with_keypoints_full_1/youtube_faces_with_keypoints_full_1',
            2: 'youtube_faces_with_keypoints_full_2/youtube_faces_with_keypoints_full_2',
            3: 'youtube_faces_with_keypoints_full_3/youtube_faces_with_keypoints_full_3',
            4: 'youtube_faces_with_keypoints_full_4/youtube_faces_with_keypoints_full_4'
        }
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get row data
        row = self.df.iloc[idx]
        
        # Use the file_path directly from our data structure
        npz_file = row['file_path']
        
        try:
            data = np.load(npz_file)
            
            # Get face image (use first frame)
            face_image = data['colorImages'][:, :, :, 0]  # Shape: (H, W, 3)
            
            # Get landmarks (use first frame)
            landmarks = data['landmarks2D'][:, :, 0]  # Shape: (68, 2)
            
            # Normalize landmarks to [0, 1] range
            img_h, img_w = face_image.shape[:2]
            landmarks[:, 0] = landmarks[:, 0] - landmarks[:, 0].min()
            landmarks[:, 1] = landmarks[:, 1] - landmarks[:, 1].min()
            landmarks[:, 0] /= landmarks[:, 0].max() if landmarks[:, 0].max() > 0 else 1
            landmarks[:, 1] /= landmarks[:, 1].max() if landmarks[:, 1].max() > 0 else 1
            
            # Convert image to tensor
            face_image = torch.from_numpy(face_image.astype(np.float32)).permute(2, 0, 1) / 255.0
            landmarks = torch.from_numpy(landmarks.astype(np.float32)).flatten()  # Flatten to 136 features
            
            # Apply transform if provided
            if self.transform:
                face_image = self.transform(face_image)
            
            # Get label
            label = torch.tensor(row['encoded_label'], dtype=torch.long)
            
            return face_image, landmarks, label
            
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            # Return dummy data if file not found
            face_image = torch.zeros((3, 64, 64), dtype=torch.float32)
            landmarks = torch.zeros(136, dtype=torch.float32)
            label = torch.tensor(0, dtype=torch.long)
            return face_image, landmarks, label
    
    def get_label_encoder(self):
        return self.label_encoder


class FaceIdentificationModel(nn.Module):
    """Neural network model for face identification combining image and landmark features."""
    
    def __init__(self, num_classes, image_size=(64, 64), landmark_features=136):
        super(FaceIdentificationModel, self).__init__()
        
        # Image branch (CNN)
        self.image_conv = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 32x32
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 16x16
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 8x8
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 4x4
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Landmark branch (MLP)
        self.landmark_mlp = nn.Sequential(
            nn.Linear(landmark_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(256 + 128, 512),  # CNN features + landmark features
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, image, landmarks):
        # Process image through CNN
        image_features = self.image_conv(image)  # B, 256, 1, 1
        image_features = image_features.view(image_features.size(0), -1)  # B, 256
        
        # Process landmarks through MLP
        landmark_features = self.landmark_mlp(landmarks)  # B, 128
        
        # Concatenate features
        combined_features = torch.cat([image_features, landmark_features], dim=1)  # B, 384
        
        # Final classification
        output = self.fusion_layer(combined_features)
        
        return output


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, landmarks, labels) in enumerate(dataloader):
        images = images.to(device)
        landmarks = landmarks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, landmarks)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%")
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on a dataset."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, landmarks, labels in dataloader:
            images = images.to(device)
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            
            outputs = model(images, landmarks)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def identify_face(model, image, landmarks, label_encoder, device='cpu'):
    """
    Identify a face using the trained model.
    
    Args:
        model: Trained model
        image: Tensor of shape (3, H, W)
        landmarks: Tensor of shape (68, 2)
        label_encoder: Label encoder used during training
        device: Device to run inference on
    
    Returns:
        person_name: Name of identified person
        confidence: Softmax confidence score
    """
    model.eval()
    
    # Convert landmarks to tensor if needed
    if isinstance(landmarks, np.ndarray):
        landmarks = torch.from_numpy(landmarks.astype(np.float32)).flatten()
    
    # Normalize landmarks
    img_h, img_w = image.shape[1], image.shape[2]
    landmarks_flat = landmarks.clone()
    landmarks_flat[::2] = landmarks_flat[::2] - landmarks_flat[::2].min()
    landmarks_flat[1::2] = landmarks_flat[1::2] - landmarks_flat[1::2].min()
    landmarks_flat[::2] /= landmarks_flat[::2].max() if landmarks_flat[::2].max() > 0 else 1
    landmarks_flat[1::2] /= landmarks_flat[1::2].max() if landmarks_flat[1::2].max() > 0 else 1
    
    # Prepare inputs
    image_batch = image.unsqueeze(0).to(device)
    landmarks_batch = landmarks_flat.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(image_batch, landmarks_batch)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get person name
    person_name = label_encoder.inverse_transform([predicted.cpu().item()])[0]
    
    return person_name, confidence.cpu().item()


def visualize_samples(dataset, num_samples=4):
    """Visualize sample face images and landmarks."""
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 2 * num_samples))
    
    for i in range(num_samples):
        image, landmarks, label_idx = dataset[i]
        
        # Get person name from label
        person_name = dataset.get_label_encoder().inverse_transform([label_idx.item()])[0]
        
        # Reshape image for display
        image_display = image.permute(1, 2, 0).cpu().numpy()
        
        # Reshape landmarks for display
        landmarks_display = landmarks.cpu().numpy().reshape(68, 2)
        
        # Denormalize landmarks
        landmarks_display[:, 0] *= image_display.shape[1]
        landmarks_display[:, 1] *= image_display.shape[0]
        
        # Plot image
        axes[i, 0].imshow(image_display)
        axes[i, 0].scatter(landmarks_display[:, 0], landmarks_display[:, 1], s=1, c='red')
        axes[i, 0].set_title(f"Face Image: {person_name}")
        axes[i, 0].axis('off')
        
        # Plot landmarks
        axes[i, 1].scatter(landmarks_display[:, 0], landmarks_display[:, 1], s=0.5, c='red')
        axes[i, 1].set_ylim(axes[i, 1].get_ylim()[::-1])  # Flip y-axis
        axes[i, 1].set_title(f"Landmarks: {person_name}")
        axes[i, 1].set_xlabel("X")
        axes[i, 1].set_ylabel("Y")
    
    plt.tight_layout()
    plt.show()


def main():
    """Main training and evaluation pipeline."""
    # Load dataset
    df = pd.read_csv('/Users/adityasirohi/BlurBuddy/Data/youtube_faces_with_keypoints_full.csv')
    
    # Filter out persons with very few samples
    person_counts = df['personName'].value_counts()
    df_filtered = df[df['personName'].isin(person_counts[person_counts >= 4].index)]
    
    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(df_filtered)}")
    print(f"Number of unique persons: {df_filtered['personName'].nunique()}")
    
    # Split dataset
    train_df, temp_df = train_test_split(df_filtered, test_size=0.3, random_state=42, stratify=df_filtered['personName'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['personName'])
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Create datasets
    data_dir = '/Users/adityasirohi/BlurBuddy/Data'
    train_dataset = FaceDataset(train_df, data_dir)
    val_dataset = FaceDataset(val_df, data_dir)
    test_dataset = FaceDataset(test_df, data_dir)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    num_classes = train_dataset.get_label_encoder().classes_.shape[0]
    model = FaceIdentificationModel(num_classes).to(device)
    
    print(f"Model created with {num_classes} classes")
    print(f"Number of parameters: {sum(p.numel_() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_acc = 0
    patience = 10
    no_improve = 0
    
    train_losses = []
    val_accs = []
    
    print("Starting training...")
    
    for epoch in range(50):
        print(f"\nEpoch {epoch + 1}/50:")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'label_encoder': train_dataset.get_label_encoder()
            }, 'face_identification_model.pth')
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    # Evaluate on test set
    checkpoint = torch.load('face_identification_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Create classification report
    label_encoder = checkpoint['label_encoder']
    class_names = label_encoder.classes_
    
    print("\nClassification Report:")
    report = classification_report(test_labels, test_preds, target_names=class_names)
    print(report)
    
    # Example usage with test data
    print("\nExample predictions:")
    for i, (image, landmarks, label) in enumerate(train_loader):
        if i >= 3:  # Show only first 3 examples
            break
        
        # Get ground truth
        gt_label = label[0].item()
        gt_name = train_dataset.get_label_encoder().inverse_transform([gt_label])[0]
        
        # Predict
        image_input = image[0]
        landmarks_input = landmarks[0].cpu().numpy().reshape(68, 2)
        pred_name, confidence = identify_face(
            model, image_input, landmarks_input, 
            train_dataset.get_label_encoder(), device
        )
        
        print(f"Ground Truth: {gt_name}")
        print(f"Predicted: {pred_name} (confidence: {confidence:.3f})")
        print(f"Correct: {'Yes' if gt_name == pred_name else 'No'}")
        print("---")


if __name__ == "__main__":
    main()

"""Test script for the face identification model."""

from face_identification_model import *
import matplotlib.pyplot as plt


def test_model_loading():
    """Test if the model can be loaded correctly."""
    print("Testing model loading...")
    
    # Load dataset
    df = pd.read_csv('/Users/adityasirohi/BlurBuddy/Data/youtube_faces_with_keypoints_full.csv')
    
    # Filter out persons with very few samples
    person_counts = df['personName'].value_counts()
    df_filtered = df[df['personName'].isin(person_counts[person_counts >= 4].index)]
    
    # Split dataset
    train_df, temp_df = train_test_split(df_filtered, test_size=0.3, random_state=42, stratify=df_filtered['personName'])
    
    # Create dataset
    data_dir = '/Users/adityasirohi/BlurBuddy/Data'
    train_dataset = FaceDataset(train_df, data_dir)
    
    # Create model
    num_classes = train_dataset.get_label_encoder().classes_.shape[0]
    model = FaceIdentificationModel(num_classes).to(device)
    
    print(f"Model created with {num_classes} classes")
    print(f"Model structure:\n{model}")
    
    # Test with sample data
    test_image = torch.zeros((32, 3, 64, 64)).to(device)  # Batch of test images
    test_landmarks = torch.zeros((32, 136)).to(device)  # Batch of test landmarks
    
    with torch.no_grad():
        output = model(test_image, test_landmarks)
        print(f"Model output shape: {output.shape}")
        print(f"Example output:\n{output[0]}")


def test_data_loading():
    """Test if data can be loaded correctly."""
    print("\nTesting data loading...")
    
    # Load dataset
    df = pd.read_csv('/Users/adityasirohi/BlurBuddy/Data/youtube_faces_with_keypoints_full.csv')
    
    # Filter out persons with very few samples
    person_counts = df['personName'].value_counts()
    df_filtered = df[df['personName'].isin(person_counts[person_counts >= 4].index)]
    
    # Split dataset
    train_df, temp_df = train_test_split(df_filtered, test_size=0.3, random_state=42, stratify=df_filtered['personName'])
    
    # Create dataset
    data_dir = '/Users/adityasirohi/BlurBuddy/Data'
    train_dataset = FaceDataset(train_df, data_dir)
    
    # Test dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Load a batch
    batch = next(iter(train_loader))
    images, landmarks, labels = batch
    
    print(f"Batch shapes:")
    print(f"Images: {images.shape}")
    print(f"Landmarks: {landmarks.shape}")
    print(f"Labels: {labels.shape}")
    
    # Print person names in this batch
    label_encoder = train_dataset.get_label_encoder()
    unique_labels = torch.unique(labels)
    person_names = label_encoder.inverse_transform(unique_labels.cpu().numpy())
    
    print(f"\\nUnique persons in batch: {len(unique_labels)}")
    for label, name in zip(unique_labels, person_names):
        print(f"Label {label.item()}: {name}")


def test_inference():
    """Test inference on a sample image."""
    print("\nTesting inference...")
    
    # Load dataset
    df = pd.read_csv('/Users/adityasirohi/BlurBuddy/Data/youtube_faces_with_keypoints_full.csv')
    
    # Filter out persons with very few samples
    person_counts = df['personName'].value_counts()
    df_filtered = df[df['personName'].isin(person_counts[person_counts >= 4].index)]
    
    # Split dataset
    train_df, temp_df = train_test_split(df_filtered, test_size=0.3, random_state=42, stratify=df_filtered['personName'])
    
    # Create dataset
    data_dir = '/Users/adityasirohi/BlurBuddy/Data'
    train_dataset = FaceDataset(train_df, data_dir)
    
    # Create model
    num_classes = train_dataset.get_label_encoder().classes_.shape[0]
    model = FaceIdentificationModel(num_classes).to(device)
    
    # Load sample data
    sample_image, sample_landmarks, sample_label = train_dataset[0]
    
    # Test identify_face function
    label_encoder = train_dataset.get_label_encoder()
    person_name, confidence = identify_face(sample_image, sample_landmarks, label_encoder, device)
    
    print(f"Identified person: {person_name}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Ground truth label: {sample_label.item()}")


def main():
    """Run all tests."""
    print("Running face identification model tests...\\n")
    
    test_model_loading()
    test_data_loading()
    test_inference()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()

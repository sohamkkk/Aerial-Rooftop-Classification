"""
Step 3: Train a Rooftop Type Classifier
=========================================
Trains a ResNet18 model (pretrained on ImageNet) to classify
individual rooftop crops into: Gable, Hip, or Flat.

Skips crops labeled as 'skip' (unclear/bad quality).
Uses stratified train/val split for balanced evaluation.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import time

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROPS_DIR = os.path.join(BASE_DIR, "Rooftop_Crops")
LABELS_FILE = os.path.join(CROPS_DIR, "labels.json")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "rooftop_classifier.pth")

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
IMG_SIZE = 128  # Resize crops to 128x128
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# Class mapping
CLASS_NAMES = ['flat', 'gable', 'hip']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RooftopDataset(Dataset):
    """Dataset for rooftop crop classification."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(is_train=True):
    """Get data transforms with augmentation for training."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),  # Rooftops can be at any orientation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def load_data():
    """Load labeled data, filtering out 'skip' labels."""
    with open(LABELS_FILE, 'r') as f:
        labels_dict = json.load(f)
    
    image_paths = []
    labels = []
    
    for crop_name, label in labels_dict.items():
        if label in CLASS_TO_IDX:  # Skip 'skip' labels
            img_path = os.path.join(CROPS_DIR, crop_name)
            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(CLASS_TO_IDX[label])
    
    return image_paths, labels


def create_model():
    """Create a ResNet18 model fine-tuned for rooftop classification."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze early layers (fine-tune only later layers)
    for name, param in model.named_parameters():
        if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    
    # Replace final FC layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, NUM_CLASSES)
    )
    
    return model.to(DEVICE)


def train_epoch(model, loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / total, correct / total, all_preds, all_labels


def main():
    print("=" * 60)
    print("ROOFTOP TYPE CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Load data
    print("\n--- Loading Data ---")
    image_paths, labels = load_data()
    print(f"Total usable labeled samples: {len(labels)}")
    
    label_counts = Counter(labels)
    for cls_name, cls_idx in CLASS_TO_IDX.items():
        print(f"  {cls_name}: {label_counts.get(cls_idx, 0)}")
    
    # Stratified train/val split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=VAL_SPLIT, 
        random_state=RANDOM_SEED, stratify=labels
    )
    
    print(f"\nTrain set: {len(train_labels)} samples")
    print(f"Val set:   {len(val_labels)} samples")
    
    # Create datasets
    train_dataset = RooftopDataset(train_paths, train_labels, 
                                    transform=get_transforms(is_train=True))
    val_dataset = RooftopDataset(val_paths, val_labels, 
                                  transform=get_transforms(is_train=False))
    
    # Weighted sampler for class imbalance
    train_label_counts = Counter(train_labels)
    weights = [1.0 / train_label_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                               sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                             shuffle=False, num_workers=2)
    
    # Create model
    print("\n--- Creating Model ---")
    model = create_model()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                       patience=3, factor=0.5)
    
    # Training loop
    print("\n--- Training ---")
    best_val_acc = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_preds, val_labels_out = evaluate(model, val_loader, criterion)
        
        scheduler.step(val_acc)
        elapsed = time.time() - start
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': CLASS_NAMES,
                'class_to_idx': CLASS_TO_IDX,
                'img_size': IMG_SIZE,
                'val_acc': best_val_acc,
                'epoch': epoch + 1
            }, MODEL_PATH)
            print(f"  ✓ New best model saved (val_acc={best_val_acc:.4f})")
    
    # Final evaluation
    print("\n--- Final Evaluation (Best Model) ---")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, final_acc, final_preds, final_labels = evaluate(model, val_loader, criterion)
    
    print(f"\nBest Validation Accuracy: {final_acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(final_labels, final_preds, 
                                 target_names=CLASS_NAMES, digits=4))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(final_labels, final_preds)
    print(f"{'':>8} {'flat':>8} {'gable':>8} {'hip':>8}")
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>8} {row[0]:>8} {row[1]:>8} {row[2]:>8}")
    
    print(f"\nModel saved to: {MODEL_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()

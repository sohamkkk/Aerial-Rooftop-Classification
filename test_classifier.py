"""
Evaluate ResNet on the Fully Manually Verified Dataset
======================================================
Loads the pre-trained `rooftop_classifier.pth` and evaluates it
specifically against every human-verified label in `labels.json`.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROPS_DIR = os.path.join(BASE_DIR, "Rooftop_Crops")
LABELS_FILE = os.path.join(CROPS_DIR, "labels.json")
MODEL_PATH = os.path.join(BASE_DIR, "model", "rooftop_classifier.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RooftopDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def main():
    print("=" * 60)
    print("EVALUATING MODEL ON 100% VERIFIED DATASET")
    print("=" * 60)

    # 1. Load model checkpoint to get exact config
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint['class_names']
    class_to_idx = checkpoint['class_to_idx']
    img_size = checkpoint['img_size']
    
    print(f"Loaded '{MODEL_PATH}'")
    print(f"Original Validation Accuracy during training: {checkpoint['val_acc']:.4f}")

    # Rebuild model structure
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, len(class_names)))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    # 2. Setup Data
    with open(LABELS_FILE, 'r') as f:
        labels_dict = json.load(f)
    
    image_paths, true_labels = [], []
    for crop_name, label in labels_dict.items():
        if label in class_to_idx: # ignores "skip"
            p = os.path.join(CROPS_DIR, crop_name)
            if os.path.exists(p):
                image_paths.append(p)
                true_labels.append(class_to_idx[label])

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = RooftopDataset(image_paths, true_labels, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Total human-verified valid crops found: {len(true_labels)}")
    print("Running inference... this may take a minute on CPU.\n")

    # 3. Predict Full Set
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Report
    print("=" * 60)
    print("FINAL CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"{'':>8} {'flat':>8} {'gable':>8} {'hip':>8}")
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>8} {row[0]:>8} {row[1]:>8} {row[2]:>8}")
    print("=" * 60)

if __name__ == '__main__':
    main()

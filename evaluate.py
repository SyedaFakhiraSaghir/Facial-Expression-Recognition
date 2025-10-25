import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import kagglehub

# 1. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 2. Download FER2013 CSV
print("Downloading FER2013 dataset...")
path = kagglehub.dataset_download("deadskull7/fer2013")
csv_path = os.path.join(path, "fer2013.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError("fer2013.csv not found in Kaggle dataset directory.")

data = pd.read_csv(csv_path)
test_df = data[data['Usage'] == 'PrivateTest']

# 3. Define Dataset Class
class FERDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = np.array(self.data.iloc[idx, 1].split(), dtype='uint8')
        image = pixels.reshape(48, 48)
        image = Image.fromarray(image)
        label = int(self.data.iloc[idx, 0])
        if self.transform:
            image = self.transform(image)
        return image, label

# 4. Define Transform
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = FERDataset(test_df, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 5. Define Model
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=len(classes)):
        super(EmotionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x

# 6. Load Model
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("best_emotion_model.pth", map_location=device))
model.eval()

# 7. Evaluate on Test Set
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(y_true, y_pred) * 100
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"\n✅ Test Accuracy: {acc:.2f}% | F1 Score: {f1:.4f}")

# 8. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - FER2013")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 9. Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

# 10. Visualize a Few Predictions
def show_predictions(dataset, preds, labels, n=6):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        img, true_label = dataset[i]
        plt.subplot(2, 3, i + 1)
        plt.imshow(img[0], cmap='gray')
        plt.title(f"True: {classes[true_label]}\nPred: {classes[preds[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_predictions(test_dataset, y_pred, y_true, n=6)

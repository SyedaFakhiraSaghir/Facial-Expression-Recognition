import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import kagglehub

# 1. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 2. Download and Load FER2013 Dataset
print("Checking for FER2013 dataset...")
path = kagglehub.dataset_download("deadskull7/fer2013")
csv_path = os.path.join(path, "fer2013.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError("fer2013.csv not found. Please check dataset structure.")

data = pd.read_csv(csv_path)
print("Dataset loaded. Total samples:", len(data))

# 3. Custom Dataset Class
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

# 4. Data Preprocessing
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Split into train/val/test
train_df = data[data['Usage'] == 'Training']
val_df = data[data['Usage'] == 'PublicTest']
test_df = data[data['Usage'] == 'PrivateTest']

train_dataset = FERDataset(train_df, transform=transform_train)
val_dataset = FERDataset(val_df, transform=transform_test)
test_dataset = FERDataset(test_df, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 5. Define CNN Model
class EmotionCNN(nn.Module):
    def __init__(self):
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
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x

# 6. Train, Validate, Test
def train_model(model, criterion, optimizer, num_epochs=15):
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100 * correct / total
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        val_acc = accuracy_score(y_true, y_pred) * 100
        val_f1 = f1_score(y_true, y_pred, average='weighted')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_emotion_model.pth")
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
    print("\nTraining Complete. Best Val Accuracy: {:.2f}%".format(best_val_acc))

# 7. Hyperparameter Tuning Example
def run_with_hyperparams(lr_values=[0.001, 0.0005], batch_sizes=[64, 128]):
    for lr in lr_values:
        for batch in batch_sizes:
            print(f"\nTraining with LR={lr}, Batch={batch}")
            model = EmotionCNN().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_model(model, criterion, optimizer, num_epochs=10)

# 8. Evaluate on Test Set
def evaluate_model(model_path="best_emotion_model.pth"):
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
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
    print(f"Test Accuracy: {acc:.2f}% | F1 Score: {f1:.4f}")

# Run Training and Testing
if __name__ == "__main__":
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, criterion, optimizer, num_epochs=15)
    evaluate_model()
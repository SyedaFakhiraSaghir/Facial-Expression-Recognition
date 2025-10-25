import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# ------------------------------------------
# Load Trained Model
# ------------------------------------------
checkpoint = torch.load("saved_models/emotion_cnn.pth", map_location=torch.device('cpu'))
classes = checkpoint['classes']

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

model = EmotionCNN()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ------------------------------------------
# Define Transform
# ------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ------------------------------------------
# User Input
# ------------------------------------------
image_path = input("Enter image path: ").strip()
if not os.path.exists(image_path):
    print("Image not found!")
    exit()

image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0)  # (1, 1, 48, 48)

with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    emotion = classes[predicted.item()]

print(f"\nPredicted Emotion: {emotion}")

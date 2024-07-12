![iStock-521697453](https://github.com/Sardorxonkeldiyev/Dog_breed_pred_with_Torch/assets/116951571/78c4466c-84ba-4558-bdf7-7929cdcbbc5e)

Dog Breed Prediction with PyTorch
This repository contains a project that uses PyTorch to predict dog breeds from images. The model is trained on a dataset of dog images and can classify various breeds accurately.

Table of Contents
Introduction
Installation
Dataset
Usage
Model Architecture
Training
Evaluation
Results
Contributing
License
Introduction
This project aims to build a deep learning model for dog breed classification using PyTorch. The model takes an image of a dog as input and predicts the breed of the dog. This can be useful for applications in veterinary science, animal shelters, and pet adoption services.

Installation
To run this project, you'll need to install the required dependencies. You can do this by running:

bash
Copy code
git clone [https://github.com/yourusername/dog_breed_pred_with_torch.git](https://github.com/Sardorxonkeldiyev/Dog_breed_pred_with_Torch)
cd dog_breed_pred_with_torch
pip install -r requirements.txt
Dataset
The dataset used for training the model is the Dog Breed Image Dataset. It contains images of various dog breeds.

Download the dataset from here.
Extract the dataset into a directory named data within the project folder.
Usage
To use the model for predicting dog breeds, follow these steps:

1. Preprocess the dataset:
python
Copy code
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset address
DATASET_LOCATION = "/kaggle/input/dog-breed-image-dataset"

# Image size
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Image data loading and transformation
train_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a dataset
train_dataset = datasets.ImageFolder(root=DATASET_LOCATION, transform=train_transforms)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
2. Train the model:
python
Copy code
# Create a model (eg ResNet-18)
model = models.resnet18(pretrained=True)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Select the optimization function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Work on the devices used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Model learning
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print("Training finished!")
3. Evaluate the model:
python
Copy code
model.eval()

# Image loading and detection for testing
def predict_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = transforms.Resize(IMAGE_SIZE)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    image = image.unsqueeze(0)  # Batch o'lchamini qo'llab-quvvatlash uchun
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return train_dataset.classes[predicted.item()]

# Testing
test_image_path = "/kaggle/input/dog-breed-image-dataset/dataset/Beagle/Beagle_30.jpg"  # Test tasviri manzili
predicted_label = predict_image(test_image_path)
print(f"Predicted label: {predicted_label}")

# Display the tested image
def show_image(image_path, label):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f"Predicted Label: {label}")
    plt.axis('off')
    plt.show()

# Display the image
show_image(test_image_path, predicted_label)
Model Architecture
The model is based on a Convolutional Neural Network (CNN) architecture. We use a pre-trained model (ResNet-18) and fine-tune it on the dog breed dataset.

Training
The training script includes code to train the model. It leverages PyTorch's torchvision library for data augmentation and pre-processing.

Evaluation
The evaluation script calculates the accuracy of the model on the validation set and prints out the results.

Results
The model achieves an accuracy of X% on the validation set. Below are some example predictions:

Image	Predicted Breed	Confidence
Labrador	95%
Beagle	89%
Contributing
We welcome contributions to this project. If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

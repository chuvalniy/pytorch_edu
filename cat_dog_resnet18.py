import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
import os, shutil
import torchvision


# # Creates two folders for classes
train_path = './data/catdog/train'
test_path = './data/catdog/test'
#
# train_dir_dogs = os.path.join(path, 'dogs')
# # os.mkdir(train_dir_dogs)
#
# train_dir_cats = os.path.join(path, 'cats')
# # os.mkdir(train_dir_cats)
#
# # Splits and copies data in created folders
# for filename in os.listdir(path):
#     if filename.split('.')[0] == 'dog':
#         shutil.copy(os.path.join(path, filename), os.path.join(train_dir_dogs, filename))
#     if filename.split('.')[0] == 'cat':
#         shutil.copy(os.path.join(path, filename), os.path.join(train_dir_cats, filename))

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Data transformations
transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

# Define hyper parameters
num_epochs = 2
learning_rate = 3e-3
batch_size = 32

# Load train dataset and split it to train/validation sets
dataset = torchvision.datasets.ImageFolder(os.path.join(train_path, 'dataset',), transform=transform)
train_set, valid_set = torch.utils.data.random_split(dataset, [20000, 5000])

# Create data loaders
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)

# Device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def show_image(inp):
#     inp = inp.numpy().transpose((1, 2, 0))
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     plt.show()
#
#
# features, labels = next(iter(train_loader))
# out = torchvision.utils.make_grid(features)
# show_image(out)


# Define resnet18 model
model = models.resnet18(pretrained=True)
num_feats = model.fc.in_features  # get num_feats from resnet18 linear layer
model.fc = nn.Linear(num_feats, 2)  # put num_feats and num_classes in layer

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


train_losses = []
valid_losses = []

# Training and Validation
for epoch in range(num_epochs):
    running_loss = 0
    running_corrects = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_lr_scheduler.step()

        running_loss += loss.item()
        if (i+1) % 200 == 0:
            n_correct = 0
            n_samples = 0
            with torch.no_grad():
                model.eval()  # Validation Mode
                for images, labels in valid_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    _, preds = torch.max(outputs, 1)
                    n_samples += labels.size(0)
                    n_correct += (preds == labels).sum().item()

            model.train() # Training Mode

            train_losses.append(running_loss/len(train_loader))
            valid_losses.append(running_loss/len(valid_loader))

            print('Epoch - [{}/{}], '.format(epoch+1, num_epochs),
                  'Train Loss - {:.3f}, '.format(train_losses[-1]),
                  'Valid Loss - {:.3f}, '.format(valid_losses[-1]),
                  'Valid Accuracy - {:.3f}'.format(n_correct/n_samples))






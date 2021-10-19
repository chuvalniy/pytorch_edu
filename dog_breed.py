import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import pandas as pd
from torchvision.io import read_image
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)


class DogBreedDataSet(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        self.img_labels['label'] = LabelEncoder().fit_transform(self.img_labels.breed)
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_data_dir = '.data/dogs_breed/train'
test_data_dir = '.data/dogs_breed/test'
csv_file = 'data/dogs_breed/labels.csv'
batch_size = 8

dataset = DogBreedDataSet(csv_file, train_data_dir, transform=transform)

train_set, test_set = torch.utils.data.random_split(dataset, [7500, 2722])
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_loaders = DataLoader(test_set, batch_size=batch_size, shuffle=False)

loaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=False)
}


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()


# inputs, classes = next(iter(loaders['train']))
# out = torchvision.utils.make_grid(inputs)
# imshow(out)

dataset_sizes = {
    'train': len(train_set),
    'test': len(test_set)
}


def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=1):
    since = time.time()

    model_best_wgts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1/num_epochs}')
        print('-'*15)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0.0

            for inputs, labels in loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_acc += torch.sum(preds == labels.data)

            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc / dataset_sizes[phase]

            print('{} loss - {:.4f}, acc - {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                model_best_wgts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('best val accuracy - {:.3f}'.format(best_acc))

    model.load_state_dict(model_best_wgts)

    path = './dog_breed_resnet18.pth'
    torch.save(model_best_wgts, path)

    return model


model_conv = torchvision.models.resnet18(pretrained=True)

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 120)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_conv.parameters(), lr=1e-3, momentum=0.92)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_model(model_conv, criterion, optimizer, lr_scheduler)








import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import numpy as np
import time
import copy

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

batch_size = 8

datasets = {
    'train': torchvision.datasets.CIFAR10(root='./data', download=False, train=True, transform=transform),
    'val': torchvision.datasets.CIFAR10(root='./data', download=False, train=False, transform=transform)
}

dataloaders = {x: DataLoader(datasets[x], shuffle=True, batch_size=batch_size)
               for x in ['train', 'val']}

datasets_sizes = {x: len(datasets[x]) for x in ['train', 'val'] for x in ['train', 'val']}
class_names = datasets['train'].classes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# print transformed images of dataset
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


inputs, classes = next(iter(dataloaders['train']))

out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, lr_scheduler, num_epochs):
    since = time.time()

    model_best_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print("-" * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # sets gradient to zero before forward pass

                # forward pass
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

            epoch_loss = running_loss / datasets_sizes[phase]
            epoch_acc = running_acc / datasets_sizes[phase]

            print('{} loss - {:.4f}, acc - {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                model_best_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('best val accuracy - {:.3f}'.format(best_acc))

    model.load_state_dict(model_best_wts)

    path = './cifar10_second_benchmark.pth'
    torch.save(model_best_wts, path)

    return model


model_conv = torchvision.models.resnet18(pretrained=True)

# for param in model_conv.parameters():
#     param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 10)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_conv.parameters(), lr=1e-3, momentum=0.9)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_model(model_conv, criterion, optimizer, lr_scheduler, num_epochs=12)


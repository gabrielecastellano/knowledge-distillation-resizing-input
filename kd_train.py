import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

batch_size=32


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, dataset_sizes, num_epochs=15, start_epoch=0):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    not_improved = 0
    epoch = start_epoch

    tb_writer = SummaryWriter('runs/resnet50_cifar100_experiment_decreasingsizes_2')

    # upsampler = torch.nn.Upsample(32).to(device)
    # upsampler.eval()

    for epoch in range(start_epoch, start_epoch+num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        val_loss = 5

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                # model.eval()
                # model.fc.train()
                loader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                loader = test_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            input_size = 0
            for batch_index, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                input_size = inputs.size(-1)

                # inputs = upsampler.forward(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=batch_index * batch_size + len(inputs),
                    total_samples=len(loader.dataset)
                ))

            if phase == 'train':
                # scheduler.step()
                scheduler.step(val_loss)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val':
                val_loss = epoch_loss
                tb_writer.add_scalar('evaluation loss', epoch_loss, epoch)
                tb_writer.add_scalar('evaluation accuracy', epoch_acc, epoch)
                tb_writer.add_scalar('input size', input_size, epoch)
            else:
                tb_writer.add_scalar('training loss', epoch_loss, epoch)
                tb_writer.add_scalar('training accuracy', epoch_acc, epoch)
                tb_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc < best_acc:
                not_improved += 1
            if phase == 'val' and epoch_acc > best_acc:
                not_improved = 0
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch % 10 == 0:
                torch.save(best_model_wts, model_file)

        print()

        if not_improved > 6:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch


def test_model(model, loader, criterion, dataset_sizes):
    since = time.time()

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for batch_index, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # upsampler = torch.nn.Upsample(32).to(device)
        # inputs = upsampler.forward(inputs)

        # forward
        # track history if only in train
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        print('Testing [{test_samples}/{total_samples}]\tLoss: {:0.4f}'.format(
            loss.item(),
            test_samples=batch_index * batch_size + len(inputs),
            total_samples=len(loader.dataset)
        ))

    test_loss = running_loss / dataset_sizes['val']
    test_acc = running_corrects.double() / dataset_sizes['val']

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


load = True
train = True
model_file = 'resnet50-cifar100'

train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
test_transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

data_dir = os.path.expanduser('~/dataset')

train_dataset = torchvision.datasets.CIFAR100(root='~/dataset', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR100(root='~/dataset', train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, num_workers=8, batch_size=batch_size)

dataset_sizes = {'train': len(train_dataset), 'val': len(test_dataset)}
class_names = train_dataset.classes

device = "cpu"  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model
resnet50 = models.resnet50(pretrained=True)
num_ftrs = resnet50.fc.in_features
# resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, 256), nn.Linear(256, 100), nn.Linear(100, 100))
resnet50.fc = nn.Linear(num_ftrs, 100)
if load:
    print("Loading model from disk...")
    resnet50.load_state_dict(torch.load(model_file))
#resnet50 = resnet50.to(device)

criterion = nn.CrossEntropyLoss()
if train:
    print("Training model...")
    e = 0
    for size in range(224, 0, -32):
        lr = 0.001
        optimizer_ft = optim.SGD(resnet50.parameters(), lr=lr, momentum=0.9)
        # optimizer_ft = optim.Adam(resnet50.parameters(), lr=0.001)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor=0.1, patience=3)

        train_transform.transforms[0] = transforms.RandomResizedCrop(size)
        test_transform.transforms[0] = transforms.Resize(size)
        train_dataset = torchvision.datasets.CIFAR100(root='~/dataset', train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR100(root='~/dataset', train=False, download=True, transform=test_transform)
        train_loader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, num_workers=8, batch_size=batch_size)

        model_ft, e = train_model(resnet50, train_loader, test_loader, criterion, optimizer_ft, exp_lr_scheduler, dataset_sizes, start_epoch=e, num_epochs=25)
        # transformers[0][0].transforms[0] = transforms.RandomResizedCrop(192)

    torch.save(model_ft.state_dict(), model_file)
else:
    print("Testing model...")
    test_model(resnet50, test_loader, criterion, dataset_sizes)

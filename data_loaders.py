import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTM(Dataset):
    '''
    Definition of MNISTM dataset
    '''
    def __init__(self, root="/home/neo/dataset/mnistm/", train=True, transform=None, target_transform=None):
        super(MNISTM, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.data, self.label = torch.load(
                os.path.join(self.root, "mnistm_pytorch_train"))
        else:
            self.data, self.label = torch.load(
                os.path.join(self.root, "mnistm_pytorch_test"))

    def __getitem__(self, index):

        data, label = self.data[index], self.label[index]
        data = Image.fromarray(data.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        # Return size of dataset
        return len(self.data)


class USPS(Dataset):
    '''
    Definition of USPS dataset
    '''
    def __init__(self, root="/home/neo/dataset/usps/", train=True, transform=None, target_transform=None):
        super(USPS, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.data, self.label = torch.load(
                os.path.join(self.root, "usps_pytorch_train"))
        else:
            self.data, self.label = torch.load(
                os.path.join(self.root, "usps_pytorch_test"))

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]
        data = Image.fromarray(data, mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return len(self.label)


# Data loader for caltech dataset
caltech_root = "/home/neo/dataset/caltech/"
Caltech = torchvision.datasets.ImageFolder(caltech_root,
                                           transform=transforms.Compose([
                                               transforms.Scale(120),
                                               transforms.CenterCrop(100),
                                               transforms.ToTensor()]))
caltech_loader = torch.utils.data.DataLoader(
    Caltech, batch_size=100, shuffle=True, num_workers=4)


# Dataset loader for office dataset
office_root = "/home/neo/dataset/office/"
Office = torchvision.datasets.ImageFolder(office_root,
                                          transform=transforms.Compose([
                                              transforms.Scale(120),
                                              transforms.CenterCrop(100),
                                              transforms.ToTensor()]))

office_loader = torch.utils.data.DataLoader(
    Office, batch_size=100, shuffle=True, num_workers=4)

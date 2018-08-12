import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class MNISTM(Dataset):

    def __init__(self, root="/home/neo/dataset/mnistm/", train=True, transform=None, target_transform=None):
        super(MNISTM, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.train_data, self.train_label = torch.load(
                os.path.join(self.root, "mnistm_pytorch_train"))
        else:
            self.test_data, self.test_label = torch.load(
                os.path.join(self.root, "mnistm_pytorch_test"))

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_label[index]
        else:
            img, target = self.test_data[index], self.test_label[index]

        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        # Return size of dataset
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class USPS(Dataset):

    def __init__(self, root="/home/neo/dataset/usps/", train=True, transform =None, target_transform=None):
        super(USPS, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.data , self.label = torch.load(os.path.join(self.root, "usps_pytorch_train"))
        else:
            self.data, self.label = torch.load(os.path.join(self.root, "usps_pytorch_test"))

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]
        data = Image.fromarray(data,mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return data, label

    def __len__(self):
        return len(self.label)


'''
import torchvision.transforms as transforms
toy = torch.utils.data.DataLoader(MNISTM(transform=transforms.Compose([
                       transforms.Resize((28,28)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),batch_size=1, shuffle=True)

for index, (data, label) in enumerate (toy):
    print(data)
    print(label)
'''
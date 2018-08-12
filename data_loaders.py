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
toy = torch.utils.data.DataLoader(USPS(transform=transforms.Compose([
                       transforms.Resize((28,28)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),batch_size=1, shuffle=True)

for index, (data, label) in enumerate (toy):
    #print(data)
    print(label)
'''
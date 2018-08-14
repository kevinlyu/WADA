import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import os

class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg()*ctx.constant
        return grad_output, None

    # pylint raise E0213 warning here
    def grad_reverse(x, constant):
        """
        Extension of grad reverse layer
        """
        return GradReverse.apply(x, constant)


class Classifier(nn.Module):
    '''
    Task Classifier
    '''
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module("conv1", nn.Conv2d())

    def forward(self, x):
'''




class Extractor(nn.Module):
    '''
    convolutional autoencoder as feature extractor
    '''

    def __init__(self):
        super(Extractor, self).__init__()
        '''
        prototype of Conv2D()
        class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)[
        '''
        # encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module("conv1", nn.Conv2d(3, 32, kernel_size=5, stride=3, padding=1))
        self.encoder.add_module("bn1", nn.BatchNorm2d(32))
        self.encoder.add_module("pool1", nn.MaxPool2d(kernel_size=2))
        self.encoder.add_module("relu1", nn.ReLU())

        self.encoder.add_module("conv2", nn.Conv2d(32, 64, kernel_size=5))
        self.encoder.add_module("bn2", nn.BatchNorm2d(64))
        #self.encoder.add_module("drop2", nn.Dropout2d())
        self.encoder.add_module("pool2", nn.MaxPool2d(kernel_size=2))
        self.encoder.add_module("relu2", nn.ReLU())

        # decoder
        self.decoder = nn.Sequential()
        self.decoder.add_module(
            "deconv1", nn.ConvTranspose2d(64, 32, kernel_size=5))
        self.decoder.add_module("relu1", nn.ReLU())
        self.decoder.add_module(
            "deconv2", nn.ConvTranspose2d(32, 16, kernel_size=5))
        self.decoder.add_module("relu2", nn.ReLU())
        self.decoder.add_module(
            "deconv3", nn.ConvTranspose2d(16, 3, kernel_size=3))
        self.decoder.add_module("output", nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z) 
        # return reconstructed data and latent feature
        return x, z


e = Extractor()
print(e)
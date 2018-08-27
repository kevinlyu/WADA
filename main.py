import os
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import util
from model import *
from data_loaders import *


'''Model Components'''
source_extractor = Autoencoder().cuda()
target_extractor = Autoencoder().cuda()
relater = Relavance(10).cuda()
classifier = Classifier(10).cuda()
discriminator = Discriminator(10).cuda()


'''Criterions'''
class_criterion = nn.CrossEntropyLoss()
# Wasserstein distance should be defined by self, to be continue
domain_criterion = nn.NLLLoss()

''' Optimizers '''
extractor_optimizer = torch.optim.Adam([{"params": source_extractor.parameters()},
                                        {"params": target_extractor.parameters()},
                                        {"params": relater.parameters()}], lr=1e-3)

classifier_optimizer = torch.optim.SGD(
    classifier.parameters(), lr=1e-3, momentum=0.9)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)


''' Dataloaders'''
source_loader = torch.utils.data.DataLoader(datasets.MNIST(
    "../datasetmnist/", train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])), batch_size=100, shuffle=True)

target_loader = torch.utils.data.DataLoader(MNISTM(
    transform=transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()
    ])), batch_size=100, shuffle=True)

''' Parameters '''
total_epoch = 200


''' Training Stage '''

for epoch in range(total_epoch):
    for index, (source, target) in enumerate(zip(source_loader, target_loader)):
        
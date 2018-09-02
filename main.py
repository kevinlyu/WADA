import os
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from util import *
from model import *
from data_loaders import *
from visualization import *

''' Parameters '''
batch_size = 250
total_epoch = 1
feature_dim = 50  # feature dimension, output size of feature extractor
d_ratio = 3  # training time of discriminator in an iteration
c_ratio = 1  # training time of classifier in an iteration
gamma = 10  # parameter for gradient penalty
weight_swd = 10.0
log_interval = 50  # interval to print loss message
save_interval = 2

''' Model Components '''
# move all networks to GPU
source_extractor = Autoencoder(encoded_dim=feature_dim).cuda()
target_extractor = Autoencoder(encoded_dim=feature_dim).cuda()
relater = Relavance(feature_dim).cuda()
classifier = Classifier(feature_dim).cuda()
discriminator = Discriminator(feature_dim).cuda()


''' Criterions '''
class_criterion = nn.NLLLoss()
relater_criterion = nn.BCELoss()

''' Optimizers '''
c_optimizer = torch.optim.Adam([{"params": source_extractor.parameters()},
                                {"params": target_extractor.parameters()},
                                {"params": classifier.parameters()}
                                ],
                               lr=1e-3)

r_optimizer = torch.optim.Adam(relater.parameters(), lr=1e-3)

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-5)


''' Dataloaders '''
source_loader = torch.utils.data.DataLoader(datasets.MNIST(
    "../datasetmnist/", train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])), batch_size=batch_size, shuffle=True)

target_loader = torch.utils.data.DataLoader(MNISTM(
    transform=transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()
    ])), batch_size=batch_size, shuffle=True)


model = WADA(source_extractor, target_extractor, classifier, relater, discriminator, source_loader, target_loader, total_epoch=100, feature_dim=feature_dim, num_classes=10)
model.train()
model.save_model()
model.visualize_by_label(dim=2)

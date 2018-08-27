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

''' Parameters '''
total_epoch = 200
feature_dim = 10

''' Model Components '''
# move all networks to GPU
source_extractor = Autoencoder().cuda()
target_extractor = Autoencoder().cuda()
relater = Relavance(feature_dim).cuda()
classifier = Classifier(feature_dim).cuda()
discriminator = Discriminator(feature_dim).cuda()


''' Criterions '''
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


''' Dataloaders '''
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


''' Training Stage '''

for epoch in range(total_epoch):
    for index, (source, target) in enumerate(zip(source_loader, target_loader)):

        source_data, source_label = source
        target_data, target_label = target

        ''' sync the batch size of source and target domain '''
        size = min((source_data.shape[0], target_data.shape[0]))
        source_data, source_label = source_data[0:size], source_label[0:size]
        target_data, target_label = target_data[0:size], target_label[0:size]
        ''' only for mnist data, expand to 3 channels '''
        source_data = source_data.expand(source_data.shape[0], 3, 28, 28)

        ''' Move data to GPU '''
        source_data, source_label = source_data.cuda(), source_label.cuda()
        target_data, target_label = target_data.cuda(), target_label.cuda()
        # print(source_data.shape)
        # print(target_data.shape)

        ''' Train extractor '''
        set_requires_gradient(source_extractor, resuires_grad=False)
        set_requires_gradient(target_extractor, resuires_grad=False)
        set_requires_gradient(discriminator, resuires_grad=True)

        with torch.no_grad():
            source_recon, source_z = source_extractor(source_data)
            target_recon, target_z = target_extractor(target_data)
            #print("source:{} \t target:{}".format(source_recon.shape, target_recon.shape))

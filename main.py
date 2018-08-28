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
total_epoch = 50
feature_dim = 10  # feature dimension
d_ratio = 3  # training time of discriminator in an iteration
c_ratio = 1  # training time of classifier in an iteration
gamma = 0.5  # parameter for gradient penalty

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
#domain_criterion = nn.NLLLoss()
relater_criterion = nn.BCELoss()

''' Optimizers '''
extractor_optimizer = torch.optim.Adam([{"params": source_extractor.parameters()},
                                        {"params": target_extractor.parameters()},
                                        {"params": classifier.parameters()}], lr=1e-4)
relater_optimizer = torch.optim.Adam(relater.parameters(), lr=1e-3)


classifier_optimizer = torch.optim.SGD(
    classifier.parameters(), lr=1e-3, momentum=0.9)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-6)


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
        total_loss = 0
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

        ''' Train relater '''
        # print("train relater")
        set_requires_gradient(relater, requires_grad=True)
        set_requires_gradient(source_extractor, requires_grad=False)
        set_requires_gradient(target_extractor, requires_grad=False)

        source_recon, source_z = source_extractor(source_data)
        target_recon, target_z = target_extractor(target_data)

        ''' set tag for source and target domains'''
        src_tag = 0
        tar_tag = 1

        ''' source loss '''
        #tags = torch.cuda.FloatTensor(source_z.size(0)).fill_(src_tag)
        tags = torch.ones(source_z.size(0)).cuda()
        src_pred = relater(source_z.detach())
        src_loss = relater_criterion(src_pred, tags)
        src_loss.backward()

        ''' target loss '''
        #tags = torch.cuda.FloatTensor(target_z.size(0)).fill_(tar_tag)
        tags = torch.zeros(target_z.size(0)).cuda()
        tar_pred = relater(target_z.detach())
        tar_loss = relater_criterion(tar_pred, tags)
        tar_loss.backward()

        relater_optimizer.zero_grad()
        r_loss = src_loss + tar_loss
        relater_optimizer.step()

        total_loss += r_loss
        ''' Train discriminator '''
        # print("train discriminator")
        set_requires_gradient(source_extractor, requires_grad=False)
        set_requires_gradient(target_extractor, requires_grad=False)
        set_requires_gradient(relater, requires_grad=False)
        set_requires_gradient(discriminator, requires_grad=True)

        with torch.no_grad():
            source_recon, source_z = source_extractor(source_data)
            target_recon, target_z = target_extractor(target_data)
            #print("source:{} \t target:{}".format(source_recon.shape, target_recon.shape))

        for _ in range(d_ratio):

            gp = gradient_penalty(discriminator, source_z, target_z)

            d_source = discriminator(source_z, 10)
            d_target = discriminator(target_z, 10)

            ''' Wasserstein-2 distance '''
            wasserstein_distance = d_source.mean() - d_target.mean()

            ''' Discriminator Loss '''
            d_loss = -wasserstein_distance + gamma * gp

            discriminator_optimizer.zero_grad()
            d_loss.backward()
            discriminator_optimizer.step()

            total_loss += d_loss

        ''' Train classfier '''
        # print("train classifier")
        set_requires_gradient(source_extractor, requires_grad=True)
        set_requires_gradient(target_extractor, requires_grad=True)
        set_requires_gradient(discriminator, requires_grad=False)

        for _ in range(c_ratio):

            source_recon, source_z = source_extractor(source_data)
            target_recon, target_z = target_extractor(target_data)

            source_preds = classifier(source_z)
            c_loss = class_criterion(source_preds, source_label)
            total_loss += c_loss
            # wasserstein_distance = classifier(source_z).mean() - classifier(target_z).mean()

            extractor_optimizer.zero_grad()
            c_loss.backward()
            extractor_optimizer.step()

        if index % 50 == 0:
            print("[Epoch{:3d}] ==> Total loss: {:.4f} \t Class loss: {:.4f} \t Relavance loss: {:.4f} \t Domain loss: {:.4f}".format(
                epoch, total_loss, c_loss, r_loss, d_loss))

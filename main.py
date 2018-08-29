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
weight_swd = 10.0

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
# domain_criterion = nn.NLLLoss()
relater_criterion = nn.BCELoss()

''' Optimizers '''
c_optimizer = torch.optim.Adam([{"params": source_extractor.parameters()},
                                {"params": target_extractor.parameters()},
                                {"params": classifier.parameters()}
                                ],
                               lr=1e-3)

r_optimizer = torch.optim.Adam(relater.parameters(), lr=1e-3)

'''
c_optimizer = torch.optim.SGD(
    classifier.parameters(), lr=1e-3, momentum=0.9)
'''
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-6)


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

        ''' Train Classifier '''
        c_optimizer.zero_grad()

        set_requires_gradient(source_extractor, requires_grad=True)
        set_requires_gradient(target_extractor, requires_grad=True)
        set_requires_gradient(discriminator, requires_grad=False)

        source_recon, source_z = source_extractor(source_data)
        target_recon, target_z = target_extractor(target_data)

        l1_src = F.l1_loss(source_recon, source_data)
        l1_tar = F.l1_loss(target_recon, target_data)
        bce_src = F.binary_cross_entropy(source_recon, source_data)
        bce_tar = F.binary_cross_entropy(target_recon, target_data)
        w2_src = weight_swd*sliced_wasserstein_distance(source_z).cuda()
        w2_tar = weight_swd*sliced_wasserstein_distance(target_z).cuda()
        wasserstein_z = source_z.mean() - target_z.mean()

        pred_src = classifier(source_z)
        c_loss = class_criterion(pred_src, source_label)
        loss = l1_src+l1_tar+bce_src+bce_tar+w2_src+w2_tar+wasserstein_z+c_loss
        #loss = l1_src+l1_tar+bce_src+bce_tar+w2_src+w2_tar+c_loss
        loss.backward()
        c_optimizer.step()
        '''
        # debug use, show accuracy of classifier
        _, pred_result = torch.max(pred_src, 1)
        c_loss = class_criterion(pred_src, source_label)
        correct = (pred_result == source_label).sum()
        accu = 100 * correct / source_data.shape[0]
        '''

        ''' Train Relater '''
        r_optimizer.zero_grad()

        set_requires_gradient(source_extractor, requires_grad=False)
        set_requires_gradient(target_extractor, requires_grad=False)
        set_requires_gradient(relater, requires_grad=True)

        with torch.no_grad():
            _, source_z = source_extractor(source_data)
            _, target_z = target_extractor(target_data)

        ''' tag 0 for source domain, tag 1 for target domain '''

        src_tag = torch.zeros(source_z.size(0)).cuda()
        src_pred = relater(source_z.detach())
        src_loss = relater_criterion(src_pred, src_tag)

        tar_tag = torch.ones(target_z.size(0)).cuda()
        tar_pred = relater(target_z.detach())
        tar_loss = relater_criterion(tar_pred, tar_tag)

        r_loss = 0.5*(src_loss + tar_loss)
        r_loss.backward()
        r_optimizer.step()

        ''' Train Discriminator '''
        d_optimizer.zero_grad()

        set_requires_gradient(source_extractor, requires_grad=False)
        set_requires_gradient(target_extractor, requires_grad=False)
        set_requires_gradient(discriminator, requires_grad=True)

        with torch.no_grad():
            ''' When training discriminator, fix parameters of feature extractors '''
            _, source_z = source_extractor(source_data)
            _, target_z = target_extractor(target_data)

        d_src = discriminator(source_z, 100)
        d_tar = discriminator(target_z, 100)

        ''' Wasserstein-2 distance with gradient penalty'''
        w2_loss = d_src.mean()-d_tar.mean()
        gp = gradient_penalty(discriminator, source_z, target_z)

        d_loss = -w2_loss + gamma * gp

        d_loss.backward()
        d_optimizer.step()

        if index % 50 == 0:
            '''
            print("[Epoch{:3d}] ==> Total loss: {:.4f} \t Class loss: {:.4f} \t Relavance loss: {:.4f} \t Domain loss: {:.4f}".format(
            epoch, total_loss, c_loss, r_loss, d_loss))
            '''
            '''
            print(
                "l1_src: {:.4f} \t l1_tar: {:.4f} \t bce_src: {:.4f} \t bce_tar: {:.4f} \t w2_src: {:.4f} \t w2_tar: {:.4f} \t, w_z: {:.4f} \t".format(l1_src, l1_tar, bce_src, bce_tar, w2_src, w2_tar, wasserstein_z))
            '''
            #print("c_loss: {:.4f} \t accuracy: {:.2f}".format(c_loss, accu))

            print("\n")
            print("R(src) {:.2f}".format(src_pred.mean()))
            print("R(tar) {:.2f}".format(tar_pred.mean()))
            print("c_loss: {:.4f}\tr_loss: {:.4f}\td_loss: {:.4f}".format(
                c_loss, r_loss, d_loss))

            #print("R(src): {:.4f}\t max: {:.4f}\t min: {:.4f}".format(test, torch.max(test), torch.min(test)))

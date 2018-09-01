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
batch_size = 1000
total_epoch = 200
feature_dim = 300  # feature dimension, output size of feature extractor
d_ratio = 3  # training time of discriminator in an iteration
c_ratio = 1  # training time of classifier in an iteration
gamma = 10  # parameter for gradient penalty
weight_swd = 10.0
log_interval = 50  # interval to print loss message

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
        # exit()
        ''' Train Classifier '''
        c_optimizer.zero_grad()

        set_requires_gradient(source_extractor, requires_grad=True)
        set_requires_gradient(target_extractor, requires_grad=True)
        set_requires_gradient(discriminator, requires_grad=False)

        source_recon, source_z = source_extractor(source_data)
        target_recon, target_z = target_extractor(target_data)
        
        # print(source_z.shape)
        # print(target_z.shape)
        # exit()

        l1_src = F.l1_loss(source_recon, source_data)
        l1_tar = F.l1_loss(target_recon, target_data)
        bce_src = F.binary_cross_entropy(source_recon, source_data)
        bce_tar = F.binary_cross_entropy(target_recon, target_data)
        w2_src = weight_swd*sliced_wasserstein_distance(source_z, embedding_dim=feature_dim).cuda()
        w2_tar = weight_swd*sliced_wasserstein_distance(target_z, embedding_dim=feature_dim).cuda()
        wasserstein_z = (source_z.mean() - target_z.mean())**2
        pred_src = classifier(source_z)
        c_loss = class_criterion(pred_src, source_label)
        loss = l1_src+l1_tar+bce_src+bce_tar+w2_src+w2_tar+c_loss+wasserstein_z
        #loss = l1_src+l1_tar+bce_src+bce_tar+w2_src+w2_tar+c_loss
        loss.backward()
        c_optimizer.step()

        # debug use, show accuracy of classifier
        _, pred_result = torch.max(pred_src, 1)

        c_loss = class_criterion(pred_src, source_label)
        correct = (pred_result == source_label).sum()
        accu = 100.0 * correct / source_data.shape[0]

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

        #print(src_pred.shape)
        #print(tar_pred.shape)
        #exit()

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

            r_src = relater(source_z)
            r_tar = relater(target_z)

        gp = gradient_penalty(discriminator, source_z, target_z)
        
        d_src = discriminator(source_z)
        d_tar = discriminator(target_z)

        #print(d_src.shape)
        #print(d_tar.shape)
        #exit()

        ''' Wasserstein-2 distance with gradient penalty'''
        w2_loss = d_src.mean()-d_tar.mean()
        

        d_loss = w2_loss + gamma * gp
        d_loss.backward()
        d_optimizer.step()

        ''' Print loss message every log_interval steps'''
        if index % log_interval == 0:
            print("[Epoch{:3d}] ==> C_loss: {:.4f}\tR_loss: {:.4f}\tD_loss: {:.4f}".format(epoch,
                                                                                           c_loss, r_loss, d_loss))

''' Concatenate source and target domain data, then plot t-SNE embedding'''
data = np.concatenate((source_z.cpu().numpy(), target_z.cpu().numpy()))
label = np.concatenate(
    (source_label.cpu().numpy(), target_label.cpu().numpy()))
visualize(data, label, dim=2, num_classes=10)

''' Save model parameters '''

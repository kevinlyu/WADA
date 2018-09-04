import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter
import numpy as np
import os
import time
from datetime import datetime
from util import *


def get_theta(embedding_dim, num_samples=50):
    theta = [w/np.sqrt((w**2).sum())
             for w in np.random.normal(size=(num_samples, embedding_dim))]
    theta = np.asarray(theta)
    return torch.from_numpy(theta).type(torch.FloatTensor)


def random_uniform(batch_size, embedding_dim):
    z = 2*(np.random.uniform(size=(batch_size, embedding_dim))-0.5)
    return torch.from_numpy(z).type(torch.FloatTensor)


def sliced_wasserstein_distance(encoded_samples, embedding_dim, distribution_fn=random_uniform, num_projections=50, p=2):

    batch_size = encoded_samples.size(0)
    z_samples = distribution_fn(batch_size, embedding_dim)
    theta = get_theta(embedding_dim, num_projections)
    encoded_samples = encoded_samples.cpu()
    proj_ae = encoded_samples.matmul(theta.transpose(0, 1))
    proj_z = z_samples.matmul(theta.transpose(0, 1))
    w_distance = torch.sort(proj_ae.transpose(0, 1), dim=1)[
        0]-torch.sort(proj_z.transpose(0, 1), dim=1)[0]

    w_distance_p = torch.pow(w_distance, p)

    return w_distance_p.mean()


def gradient_penalty(critic, h_s, h_t):

    alpha = torch.rand(h_s.size(0), 1).cuda()
    difference = h_t-h_s
    interpolates = h_s + (alpha * difference)
    """
    Reference
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/wdgrl.py
    """
    interpolates.requires_grad_()
    # interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()
    # interpolates = interpolates.view(-1, 100)
    preds = critic(interpolates)
    gradients = grad(preds, interpolates, grad_outputs=torch.ones_like(
        preds), retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm-1)**2).mean()
    return gp


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


class Extractor(nn.Module):
    ''' Feature extractor '''

    def __init__(self, in_channels=16, lrelu_slope=0.2, encoded_dim=100):
        super(Extractor, self).__init__()

        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope

        self.encoded_dim = encoded_dim

        self.extract = nn.Sequential(
            nn.Conv2d(3, self.in_channels*1, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(self.lrelu_slope),
            nn.Conv2d(self.in_channels*1, self.in_channels *
                      4, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(self.lrelu_slope)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.in_channels*4*7*7, self.encoded_dim),
            nn.ReLU()
        )

    def forward(self, x):
        z = self.extract(x)
        z = z.view(z.shape[0], self.in_channels*4*7*7)
        z = self.fc(z)

        return z


class Autoencoder(nn.Module):

    def __init__(self, in_channels=16, lrelu_slope=0.2, fc_dim=128, encoded_dim=100):
        super(Autoencoder, self).__init__()

        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope
        self.fc_dim = fc_dim
        self.encoded_dim = encoded_dim

        # encoder part
        self.encoder = nn.Sequential(
            nn. Conv2d(3, self.in_channels*1, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*1, self.in_channels *
                      1, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.MaxPool2d(kernel_size=2, padding=0),
            nn.Conv2d(self.in_channels*1, self.in_channels *
                      2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*2, self.in_channels *
                      2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.MaxPool2d(kernel_size=2, padding=0),
            nn.Conv2d(self.in_channels*2, self.in_channels *
                      4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*4, self.in_channels *
                      4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.MaxPool2d(kernel_size=2, padding=1)
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.in_channels*4*4*4, self.fc_dim),
            nn.ReLU(),
            nn.Linear(self.fc_dim, self.encoded_dim)
        )

        # decoder part
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.encoded_dim, self.fc_dim),
            nn.Linear(self.fc_dim, self.in_channels*4*4*4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels*4, self.in_channels *
                      4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*4, self.in_channels *
                      4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels*4, self.in_channels *
                      4, kernel_size=3, padding=0),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*4, self.in_channels *
                      4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels*4, self.in_channels *
                      2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*2, self.in_channels *
                      2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*2, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(-1, self.in_channels*4*4*4)
        z = self.encoder_fc(z)

        x = self.decoder_fc(z)
        x = x.view(-1, 4*self.in_channels, 4, 4)
        x = self.decoder(x)
        x = F.sigmoid(x)

        return x, z


class Classifier(nn.Module):
    """
    Task Classifier
    """

    def __init__(self, in_dim):
        super(Classifier, self).__init__()

        self.in_dim = in_dim

        self.fc1 = nn.Linear(self.in_dim, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 75)
        self.bn2 = nn.BatchNorm1d(75)
        self.fc3 = nn.Linear(75, 50)
        self.bn3 = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        logits = F.relu(self.bn1(self.fc1(x)))
        logits = self.fc2(F.dropout(logits))
        logits = F.relu(self.bn2(logits))
        logits = self.fc3(F.dropout(logits))
        logits = F.relu(self.bn3(logits))
        logits = self.fc4(logits)

        # for BCE loss
        # return F.softmax(logits, 1)

        # for NLL loss
        return F.log_softmax(logits, 1)


class Discriminator(nn.Module):

    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(self.in_dim, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 25)
        self.bn2 = nn.BatchNorm1d(25)
        self.fc3 = nn.Linear(25, 2)

    def forward(self, x):
        # x = GradReverse.grad_reverse(x, constant)
        logits = F.relu(self.bn1(self.fc1(x)))
        logits = F.relu(self.bn2(self.fc2(logits)))
        logits = F.dropout(logits)
        logits = F.log_softmax(self.fc3(logits), 1)

        return logits


class Relavance(nn.Module):

    """
    Relanvance network to conduct partial transfer
    """

    def __init__(self, in_dim):
        super(Relavance, self).__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(self.in_dim, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 75)
        self.bn2 = nn.BatchNorm1d(75)
        self.fc3 = nn.Linear(75, 50)
        self.bn3 = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50, 25)
        self.bn4 = nn.BatchNorm1d(25)
        self.fc5 = nn.Linear(25, 1)

    def forward(self, x):
        logits = F.relu(self.bn1(self.fc1(x)))
        logits = F.relu(self.bn2(self.fc2(logits)))
        logits = F.relu(self.bn3(self.fc3(logits)))
        logits = F.relu(self.bn4(self.fc4(logits)))
        logits = F.sigmoid(self.fc5(logits))
        # logits = F.dropout(logits)

        return logits


class WADA:
    """ WADA model """

    def __init__(self, src_extractor, tar_extractor, classifier, relater, discriminator, src_loader, tar_loader, total_epoch=20, img_size=28, feature_dim=150, num_classes=10):
        """ Parameters """
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.total_epoch = total_epoch
        self.log_interval = 10
        self.img_size = img_size
        self.weight_swd = 10
        self.gamma = 0.5
        """ Components of  WADA"""
        self.src_extractor = src_extractor
        self.tar_extractor = tar_extractor
        self.classifier = classifier
        self.relater = relater
        self.discriminator = discriminator

        """ Dataloader """
        self.src_loader = src_loader
        self.tar_loader = tar_loader

        """ Criterions """
        self.c_criterion = nn.NLLLoss()
        self.r_criterion = nn.BCELoss()

        """ Optimizers """
        self.c_opt = torch.optim.Adam([{"params": self.src_extractor.parameters()},
                                       {"params": self.tar_extractor.parameters()},
                                       {"params": self.classifier.parameters()}], lr=1e-4)
        self.r_opt = torch.optim.Adam(self.relater.parameters(), lr=1e-4)
        self.d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-6)

    def train(self):
        """ Train WADA """
        print("[start training]")

        for epoch in range(self.total_epoch):
            for index, (src, tar) in enumerate(zip(self.src_loader, self.tar_loader)):
                src_data, src_label = src
                tar_data, tar_label = tar

                size = min(src_data.shape[0], tar_data.shape[0])
                src_data, src_label = src_data[0:size], src_label[0:size]
                tar_data, tar_label = tar_data[0:size], tar_label[0:size]

                """ For MNIST data, expand number of channel to 3 """
                if src_data.shape[1] != 3:
                    src_data = src_data.expand(
                        src_data.shape[0], 3, self.img_size, self.img_size)

                src_data, src_label = src_data.cuda(), src_label.cuda()
                tar_data, tar_label = tar_data.cuda(), tar_label.cuda()

                """ Train Classifier """
                self.c_opt.zero_grad()

                '''
                src_rec, src_z = self.src_extractor(src_data)
                tar_rec, tar_z = self.src_extractor(tar_data)
                '''

                src_z = self.src_extractor(src_data)
                tar_z = self.src_extractor(tar_data)

                '''
                l1_src = F.l1_loss(src_rec, src_data)
                l1_tar = F.l1_loss(tar_rec, tar_data)
                bce_src = F.binary_cross_entropy(src_rec, src_data)
                bce_tar = F.binary_cross_entropy(tar_rec, tar_data)
                '''

                '''
                w2_src = self.weight_swd * \
                    sliced_wasserstein_distance(
                        src_z, embedding_dim=src_z.shape[1]).cuda()
                w2_tar = self.weight_swd * \
                    sliced_wasserstein_distance(
                        tar_z, embedding_dim=tar_z.shape[1]).cuda()
                '''
                w_z = (src_z.mean()-tar_z.mean())**2

                pred_label = self.classifier(src_z)
                c_loss = self.c_criterion(pred_label, src_label)
                '''
                c_loss = c_loss + w2_src + w2_tar + l1_src+l1_tar+bce_src+bce_tar+10 * \
                    (self.discriminator(src_z).mean() -
                     self.discriminator(tar_z).mean())**2
                '''

                c_loss = c_loss + w_z + \
                    self.discriminator(src_z).mean() - \
                    self.discriminator(tar_z).mean()
                c_loss.backward()
                self.c_opt.step()

                """ Train Relater"""
                self.r_opt.zero_grad()

                with torch.no_grad():
                    src_z = self.src_extractor(src_data)
                    tar_z = self.src_extractor(tar_data)

                src_tag = torch.zeros(src_z.size(0)).cuda()
                src_pred = self.relater(src_z)
                src_loss = self.r_criterion(src_pred, src_tag)

                tar_tag = torch.ones(tar_z.size(0)).cuda()
                tar_pred = self.relater(tar_z)
                tar_loss = self.r_criterion(tar_pred, tar_tag)

                r_loss = src_loss + tar_loss
                r_loss.backward()
                self.r_opt.step()

                """ Train Discriminator """
                for _ in range(5):
                    self.d_opt.zero_grad()

                    with torch.no_grad():
                        '''
                        _, src_z = self.src_extractor(src_data)
                        _, tar_z = self.tar_extractor(tar_data)
                        '''
                        src_z = self.src_extractor(src_data)
                        tar_z = self.src_extractor(tar_data)

                        src_r = self.relater(src_z)
                        tar_r = self.relater(tar_z)

                    gp = gradient_penalty(self.discriminator, src_z, tar_z)

                    d_src = self.discriminator(src_z)
                    d_tar = self.discriminator(tar_z)

                    w2_loss = (d_src.mean()-d_tar.mean())**2
                    d_loss = -w2_loss + self.gamma * gp

                    d_loss.backward()
                    self.d_opt.step()

                if index % self.log_interval == 0:
                    print("[Epoch{:3d}] ==> C_loss: {:.4f}\tR_loss: {:.4f}\tD_loss: {:.4f}".format(epoch,
                                                                                                   c_loss, r_loss, d_loss))

    def test(self):
        """ Test WADA """
        print("[start testing]")

        self.src_extractor.eval()
        self.tar_extractor.eval()
        self.classifier.eval()
        self.relater.eval()
        self.discriminator.eval()

    def save_model(self, save_root="./saved_model/"):

        folder = datetime.now().strftime("%Y-%m-%d_%H-%M")
        save_dir = os.path.join(save_root, folder)

        try:
            os.stat(save_dir)
        except:
            os.mkdir(save_dir)

        torch.save(self.classifier, os.path.join(save_dir, "WADA" + "_C.pkl"))
        torch.save(self.relater, os.path.join(save_dir, "WADA" + "_R.pkl"))
        torch.save(self.discriminator, os.path.join(
            save_dir, "WADA" + "_D.pkl"))

    def load_model(self, folder, save_root="./saved_model/"):

        load_dir = os.path.join(save_root, folder)
        self.classifier.load_state_dict(
            torch.load(load_dir, "WADA" + "_C.pkl"))
        self.relater.load_state_dict(
            torch.load(load_dir, "WADA" + "_R.pkl"))
        self.discriminator.load_state_dict(
            torch.load(load_dir, "WADA" + "_D.pkl"))

    def visualize(self, dim):
        self.src_extractor.cpu()
        self.tar_extractor.cpu()

        self.src_extractor.eval()
        self.tar_extractor.eval()

        src_data = torch.FloatTensor()
        src_label = torch.LongTensor()

        for index, src in enumerate(self.src_loader):
            data, label = src
            src_data = torch.cat((src_data, data))
            src_label = torch.cat((src_label, label))

        tar_data = torch.FloatTensor()
        tar_label = torch.IntTensor()

        for index, tar in enumerate(self.tar_loader):
            data, label = tar
            tar_data = torch.cat((tar_data, data))
            tar_label = torch.cat((tar_label, label))

        if src_data.shape[1] != 3:
            src_data = src_data.expand(
                src_data.shape[0], 3, self.img_size, self.img_size)

        src_data, src_label = src_data[0:1000], src_label[0:1000]
        tar_data, tar_label = tar_data[0:1000], tar_label[0:1000]

        src_z = self.src_extractor(src_data)
        tar_z = self.src_extractor(tar_data)

        data = np.concatenate(
            (src_z.detach().numpy(), tar_z.detach().numpy()))
        label = np.concatenate(
            (src_label.numpy(), tar_label.numpy()))

        tsne = TSNE(n_components=dim, verbose=1,
                    init="pca", perplexity=40, n_iter=3000)

        embedding = tsne.fit_transform(data)

        embedding_max, embedding_min = np.max(
            embedding, 0), np.min(embedding, 0)
        embedding = (embedding-embedding_min)/(embedding_max-embedding_min)

        if dim == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            colors = cm.rainbow(np.linspace(0.0, 1.0, self.num_classes))

            xx = embedding[:, 0]
            yy = embedding[:, 1]

            for i in range(self.num_classes):
                ax.scatter(xx[label == i], yy[label == i],
                           color=colors[i], s=10)

            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')
            plt.legend(loc='best', scatterpoints=1, fontsize=5)
            plt.savefig("TSNE_Label_2D.pdf", format='pdf', dpi=600)
            plt.show()
            plt.close("all")

            src_tag = torch.zeros(src_z.size(0))
            tar_tag = torch.ones(tar_z.size(0))
            tag = np.concatenate((src_tag.numpy(), tar_tag.numpy()))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            colors = cm.rainbow(np.linspace(0.0, 1.0, self.num_classes))

            xx = embedding[:, 0]
            yy = embedding[:, 1]

            for i in range(2):
                ax.scatter(xx[tag == i], yy[tag == i],
                           color=cm.bwr(i/1.), s=10)

            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')
            plt.legend(loc='best', scatterpoints=1, fontsize=5)
            plt.savefig("TSNE_Domain_2D.pdf", format='pdf', dpi=600)
            plt.show()
            plt.close("all")

        elif dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            colors = cm.rainbow(np.linspace(0.0, 1.0, self.num_classes))

            xx = embedding[:, 0]
            yy = embedding[:, 1]
            zz = embedding[:, 2]

            for i in range(self.num_classes):
                ax.scatter(xx[label == i], yy[label == i],
                           zz[label == i], color=colors[i], s=10)

            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.zaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')
            plt.legend(loc='best', scatterpoints=1, fontsize=5)
            plt.savefig("TSNE_Label_3D.pdf", format='pdf', dpi=600)
            plt.show()
            plt.close("all")

            src_tag = torch.zeros(src_z.size(0))
            tar_tag = torch.ones(tar_z.size(0))
            tag = np.concatenate((src_tag.numpy(), tar_tag.numpy()))

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            xx = embedding[:, 0]
            yy = embedding[:, 1]
            zz = embedding[:, 2]

            for i in range(2):
                ax.scatter(xx[tag == i], yy[tag == i],
                           zz[tag == i], color=cm.bwr(i/1.), s=10)

            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.zaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')
            plt.legend(loc='best', scatterpoints=1, fontsize=5)
            plt.savefig("TSNE_Domain_3D.pdf", format='pdf', dpi=600)
            plt.show()
            plt.close("all")
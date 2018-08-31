import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import grad
import numpy as np
import os

feature_dim = 10
random_z_size = 100


def get_theta(embedding_dim, num_samples=50):
    theta = [w/np.sqrt((w**2).sum())
             for w in np.random.normal(size=(num_samples, embedding_dim))]
    theta = np.asarray(theta)
    return torch.from_numpy(theta).type(torch.FloatTensor)


def random_uniform(batch_size):
    z = 2*(np.random.uniform(size=(batch_size, random_z_size))-0.5)
    return torch.from_numpy(z).type(torch.FloatTensor)


def sliced_wasserstein_distance(encoded_samples, distribution_fn=random_uniform, num_projections=50, p=2):

    batch_size = encoded_samples.size(0)
    z_samples = distribution_fn(batch_size)
    embedding_dim = z_samples.size(1)

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
    '''
    Reference
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/wdgrl.py
    '''
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()
    preds = critic(interpolates, h_s.shape[1])
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


class SAE:

    def __init__(self, autoencoder, optimizer, distribution_fn, num_projections=50, p=2, weight_swd=10):
        self.model = autoencoder
        self.optimizer = optimizer
        self.distribution_fn = distribution_fn
        self.embedding_dim = self.model.encoded_dim
        self.num_projections = num_projections
        self.p = p
        self.weight_swd = weight_swd

    def train(self, x):
        self.optimizer.zero_grad()

        recon_x, z = self.model(x)

        l1 = F.l1_loss(recon_x, x)
        bce = F.binary_cross_entropy(recon_x, x)

        recon_x = recon_x.cpu()

        w2 = float(self.weight_swd)*sliced_wasserstein_distance(z,
                                                                self.distribution_fn, self.num_projections, self.p)
        w2 = w2.cuda()
        loss = l1+bce+w2

        loss.backward()
        self.optimizer.step()

        return {'loss': loss, 'bce': bce, 'l1': l1, 'w2': w2, 'encode': z, 'decode': recon_x}

    def test(self, x):
        self.optimizer.zero_grad()
        recon_x, z = self.model(x)

        l1 = F.l1_loss(recon_x, x)
        bce = F.binary_cross_entropy(recon_x, x)
        recon_x = recon_x.cpu()
        w2 = float(self.weight_swd)*sliced_wasserstein_distance(z,
                                                                self.distribution_fn, self.num_projections, self.p)

        w2 = w2.cuda()
        loss = l1+bce+w2

        return {'loss': loss, 'bce': bce, 'l1': l1, 'w2': w2, 'encode': z, 'decode': recon_x}


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

            nn.AvgPool2d(kernel_size=2, padding=0),
            nn.Conv2d(self.in_channels*1, self.in_channels *
                      2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*2, self.in_channels *
                      2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.AvgPool2d(kernel_size=2, padding=0),
            nn.Conv2d(self.in_channels*2, self.in_channels *
                      4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*4, self.in_channels *
                      4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.AvgPool2d(kernel_size=2, padding=1)
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
    '''
    Task Classifier
    '''

    def __init__(self, in_dim):
        super(Classifier, self).__init__()

        self.in_dim = in_dim

        self.fc1 = nn.Linear(1*self.in_dim*self.in_dim, 100)
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
    '''
    Domain discrimiator
    '''
    '''
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(1*self.in_dim*self.in_dim, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        logits = F.relu(self.bn1(self.fc1(x)))
        logits = F.log_softmax(self.fc2(logits), 1)
        return logits

    '''

    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(1*self.in_dim*self.in_dim, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 25)
        self.fc3 = nn.Linear(25, 2)

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        logits = F.relu(self.bn1(self.fc1(x)))
        logits = F.relu(self.fc2(logits))
        logits = F.dropout(logits)
        logits = F.log_softmax(self.fc3(logits), 1)
        #logits = F.softmax(self.fc2(logits), 1)

        return logits


class Relavance(nn.Module):

    '''
    Relanvance network to conduct partial transfer
    '''

    def __init__(self, in_dim):
        super(Relavance, self).__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(1*self.in_dim*self.in_dim, 100)
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
        logits = F.dropout(logits)

        return logits

import os
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import util
import model


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


mnist = torch.utils.data.DataLoader(datasets.MNIST(
    "../dataset/mnist/", train=True, download=True,
    transform=transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])), batch_size=500, shuffle=True)


model = Extractor().cuda()
# print(model)
criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)


total_epoch = 100

for epoch in range(total_epoch):

    for index, data in enumerate(mnist):
        img, _ = data
        img = Variable(img.cuda())

        img_gen = model(img)
        loss = criterion(img_gen, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, total_epoch, loss.data[0]))
    '''
    if epoch % 10 == 0:
        pic = to_img(img_gen.cpu().data)
        save_image(pic, '../dc_img/image_{}.png'.format(epoch))
    '''

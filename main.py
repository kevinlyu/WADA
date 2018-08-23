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


'''
mnist = torch.utils.data.DataLoader(datasets.MNIST(
    "../dataset/mnist/", train=True, download=True,
    transform=transforms.Compose([
        #transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])), batch_size=128, shuffle=True)


model = Extractor().cuda()
# print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-3)


total_epoch = 100

for epoch in range(total_epoch):

    for index, data in enumerate(mnist):
        img, _ = data
        img = Variable(img).cuda()

        img_gen, _ = model(img)

        loss = criterion(img_gen, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, total_epoch, loss.data[0]))
    
    if epoch % 10 == 0:
        pic = to_img(img_gen.cpu().data)
        save_image(pic, '../dc_img/image_{}.png'.format(epoch))
    
'''
mnist = torch.utils.data.DataLoader(datasets.MNIST("./mnist/", train=True, download=True,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor()
                                                   ])), batch_size=128, shuffle=True)

ae = Autoencoder().cuda()
optimizer = torch.optim.Adam(ae.parameters())
total_epoch = 1
trainer = SAE(ae, optimizer, random_uniform, num_projections=50)
ae.train()


for epoch in range(total_epoch):

    for index, (img, label) in enumerate(mnist):
        img = img.cuda()
        #img = img.expand(img.data.shape[0], 3, 28, 28)
        batch_result = trainer.train(img)
        if (index+1) % 10 == 0:
            print("loss: {:.4f} \t l1:{:.4f} \t bce:{:.4f} \t w2:{:.4f}".format(
                batch_result["loss"], batch_result["l1"], batch_result["bce"], batch_result["w2"]))

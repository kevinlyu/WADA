import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pickle
import os 

def process_mnistm(mnistm_path = "/home/neo/dataset/mnistm/",mnist_path = "/home/neo/dataset/mnist/"):
    with open(os.path.join(mnistm_path, "mnistm.pkl"), "rb") as f:
        mnistm_data = pickle.load(f, encoding="bytes")
    
    #load content of pickle file into pytorch tensor (data only)
    mnistm_train_data = torch.ByteTensor(mnistm_data[b'train'])
    mnistm_test_data = torch.ByteTensor(mnistm_data[b'test'])
    
    #load label of mnist dataset
    mnist_train_labels = datasets.MNIST(root=mnist_path, train=True,download=True).train_labels
    mnist_test_labels = datasets.MNIST(root=mnist_path,train=False,download=True).test_labels
    
    #combine (data, label)
    training_set = (mnistm_train_data, mnist_train_labels)
    test_set = (mnistm_test_data, mnist_test_labels)
    
    #save mnist data with (data, label)
    with open(os.path.join(mnistm_path, "mnistm_pytorch_train"), "wb") as f:
        torch.save(training_set, f)
        
    with open(os.path.join(mnistm_path, "mnistm_pytorch_test"), "wb") as f:
        torch.save(test_set, f)
    
    print("Done!")


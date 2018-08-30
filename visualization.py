from sklearn.manifold import TSNE
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import scipy

def save_feature_to_mat(feature, label):
    ''' save (feature, label) to .mat format '''
    ''' [Optional] use Matlab to plot TSNE result '''


def plot_tsne(data, label, domain):


def visualize(source_data, target_data, source_label, target_label, domain, title="TSNE", img_name="TSNE.png"):
    ''' Vsualize the scatter of t-SNE dimension reduction'''

    print("t-SNE processing")
    start_time = time.time()

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

    ''' Concatenate all data / label of two domains'''
    data = np.concatenate(source_data, target_data)
    label = np.concatenate(source_label, target_label)

    ''' domain tags, 0 for source , 1 for target '''
    source_tag = torch.ones(source_data.size(0))
    target_tag = torch.zeros(target_data.size(0))
    tag = np.concatenate(source_tag, target_tag)

    embedding = tsne.fit_transform(data, label)

    print("t-SNE used: {} seconds".format(time.time()-start_time))
    plot_tsne(data, label, tag)

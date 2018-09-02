from sklearn.manifold import TSNE
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter
import scipy.io as sio
import os
from mpl_toolkits.mplot3d import Axes3D

def save_feature_to_mat(embedding, label, dim, path="./mat/"):
    ''' save (embedding, label) to .mat format, can be load with python or matlab '''
    try:
        os.stat(path)
    except:
        os.mkdir(path)

    if dim == 2:
        sio.savemat(os.path.join(path, "feature_2D.mat"), {
                    "dim0": embedding[:, 0], "dim1": embedding[:, 1]})

    elif dim == 3:
        sio.savemat(os.path.join(path, "feature_2D.mat"), {
                    "dim0": embedding[:, 0], "dim1": embedding[:, 1], "dim2": embedding[:, 2]})

    print(".mat file saved.")


def plot_tsne(embedding, label, dim, num_classes=10, img_name="tsne.pdf"):
    ''' Normalize embedding data '''
    embedding_max, embedding_min = np.max(embedding, 0), np.min(embedding, 0)
    embedding = (embedding-embedding_min)/(embedding_max-embedding_min)

    print("Plotting t-SNE")

    ''' Plot according given dimension '''
    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

        xx = embedding[:, 0]
        yy = embedding[:, 1]
        zz = embedding[:, 2]

        for i in range(num_classes):
            ax.scatter(xx[label == i], yy[label == i],
                       zz[label == i], color=colors[i], s=10)

        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.zaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
        plt.legend(loc='best', scatterpoints=1, fontsize=5)
        plt.savefig(img_name, format='pdf', dpi=600)
        plt.show()

    elif dim == 2:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

        xx = embedding[:, 0]
        yy = embedding[:, 1]

        for i in range(num_classes):
            ax.scatter(xx[label == i], yy[label == i], color=colors[i], s=10)

        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
        plt.legend(loc='best', scatterpoints=1, fontsize=5)
        plt.savefig(img_name, format='pdf', dpi=600)
        plt.show()


def visualize(data, label, dim, num_classes=10, title="TSNE", img_name="TSNE.png"):
    ''' Vsualize the scatter of t-SNE dimension reduction'''

    print("t-SNE processing")
    start_time = time.time()
    '''
    tsne = TSNE(n_components=dim, verbose=1,
                init="pca", perplexity=40, n_iter=3000)
    '''
    tsne = TSNE(n_components=dim)
    embedding = tsne.fit_transform(data)

    print("t-SNE used: {} seconds".format(time.time()-start_time))

    plot_tsne(embedding, label, dim=dim,
              num_classes=num_classes, img_name=img_name)

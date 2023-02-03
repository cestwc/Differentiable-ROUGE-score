import io
import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('WebAgg') 

import matplotlib.pyplot as plt

def plot_trend(mat, title, xticks, yticks, xlabel, ylabel):
    figure = plt.figure(figsize=(10, 8))
    plt.imshow(mat.T, interpolation='nearest', cmap=plt.cm.Blues, vmin=0., vmax=1.0)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(class_names))
    plt.xticks(np.arange(len(xticks)), xticks)#, rotation=45)
    plt.yticks(np.arange(len(yticks)), np.around(yticks, decimals=2))

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(mat.astype('float'), decimals=4)

    # Use white text if squares are dark; otherwise black.
    threshold = mat.max() / 2.
    for i, j in itertools.product(range(len(xticks)), range(len(yticks))):
        color = "white" if mat[i, j] > threshold else "black"
        plt.text(i, j, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
    return figure


import gramgen
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
g = gramgen.GramGenerateLoss(reduction = 'intersection', ignore_index = 1).to(device)

torch.manual_seed(1234)
x = torch.rand(2, 10, 19).to(device)

y = torch.randint(2, 10, (2, 19)).to(device)

y[:, 8:] = 1

out = g(x, y)
print(out.shape)
plot_trend(out[0].squeeze().cpu().numpy(), 'intersection', y[0].numpy(), x[0].argmax(0).numpy(), 'y', r'$\hat{y}$')
plt.show()
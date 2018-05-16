import pickle
import numpy as np
import os
from scipy.misc import imread

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(filepath):
    """load all of cifar"""
    images = []
    labels = []
    for b in range(1,6):
        f = os.path.join(file, 'data_batch_%d' %b )
        X, Y = load_CIFAR_batch(f)
        images.append(X)
        labels.append(Y)
    Xtr = np.concatenate(images)
    Ytr = np.concatenate(labels)
    del X,Y
    Xte, Yte = load_CIFAR_batch(os.path.join(f, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

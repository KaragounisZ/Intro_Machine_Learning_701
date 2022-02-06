import numpy as np
import os
import perceptron
from scipy.io import loadmat

def get_accuracy(X, Y, w):
    correct = 0
    for j in range(X.shape[0]):
        if perceptron.perceptron_pred(X[j], w) == Y[j]:
            correct += 1
    return correct * 1. / Y.shape[0]

def load_simple_dataset(data_dir):
    filename = "simple_dataset"
    X_filename = os.path.join(data_dir, "{}_X.npy".format(filename))
    Y_filename = os.path.join(data_dir, "{}_Y.npy".format(filename))
    X = np.load(X_filename)
    Y = np.load(Y_filename)
    return X, Y

def load_mnist(data_dir, sub_set):
    filename = "perceptron_{}".format(sub_set)
    data = loadmat(os.path.join(data_dir, filename))
    data = data[sub_set][0][0]
    X = data[0].transpose()
    Y = data[1].astype("int32")[0]
    np.save(os.path.join(data_dir, filename + "_X.npy"), X)
    np.save(os.path.join(data_dir, filename + "_Y.npy"), Y)
    return X, Y


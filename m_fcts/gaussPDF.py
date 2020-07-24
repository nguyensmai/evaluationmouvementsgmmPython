import numpy as np


def gaussPDF(Data, Mu, Sigma):
    nbVar, nbData = Data.shape
    Data = Data.T - Mu.T
    prob = np.sum(np.dot(Data, np.linalg.inv(Sigma)) * Data, axis=1, keepdims=True)

    prob = np.exp(-0.5*prob) / np.sqrt((2*np.pi)**nbVar * np.abs(np.linalg.det(Sigma)) + np.finfo(float).tiny)
    return prob

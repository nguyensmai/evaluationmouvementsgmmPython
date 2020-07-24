import math
import numpy as np
from functions.basic_functions import *


def logmap(x, mu):
    """
    :return: a vertical vertor
    """
    A = np.array([[1], [0], [0], [0]])
    if np.linalg.norm(mu - A) < 1e-6:
        Q = np.identity(4)
    else:
        Q = QuatMatrix(mu)
    u = logfct(np.dot(Q.T, x))
    return u


def logfct(x):
    scale = acoslog(x[0, :]) / np.sqrt(1 - x[0, :] ** 2)
    scale = np.where(np.isnan(scale), 1, scale)
    Log = np.array([x[1, :] * scale, x[2, :] * scale, x[3, :] * scale])
    return Log


def expmap(u, mu):
    return np.dot(QuatMatrix(mu), expfct(u))


def expfct(u):
    normv = np.sqrt(u[0, :] ** 2 + u[1, :] ** 2 + u[2, :] ** 2)
    Exp = np.array([np.cos(normv), u[0, :] * np.sin(normv) / normv, u[1, :] * np.sin(normv) / normv,
                    u[2, :] * np.sin(normv) / normv])
    ind = np.where(normv < 1e-16)[0]
    for i in ind:
        Exp[:, ind:ind + 1] = np.array([[1], [0], [0], [0]])
    return Exp

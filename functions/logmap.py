import math
import numpy as np
from functions.basic_functions import QuatMatrix
from functions.basic_functions import acoslog


def logmap(x, mu):
    A = np.array([[1],[0],[0],[0]])
    if np.linalg.norm(mu - A) < 1e-6:
        Q = np.identity(4)
    else:
        Q = QuatMatrix(mu,axis=1)
    u = logfct(np.dot(Q.T, x))
    return u

def logfct(x):
    scale = np.array([acoslog(b) / math.sqrt(1-b**2) for b in x[0, :]])
    for i in range(np.size(scale)):
        if np.isnan(scale[i]):
            scale[i] = 1
    Log = np.array([x[1,:]*scale, x[2,:]*scale, x[3,:]*scale])
    return Log

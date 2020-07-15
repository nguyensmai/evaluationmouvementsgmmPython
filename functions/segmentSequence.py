import math
import numpy as np
from functions.basic_functions import logmap
from functions.peakdet import *

def segmentSequenceKeyPose( _in, ws, thres ):
    variation = np.full([1, _in.shape[1]], np.nan)
    for t in range(math.ceil(ws / 2), _in.shape[1] - math.floor(ws / 2) + 1):
        sigma = 0
        for w in np.arange(-math.floor(ws/2),math.floor(ws/2)+1):
            for d in range(int(_in.shape[0]/4)):
                muMan = _in[d*4: d*4+4, t-1:t]
                sigma = sigma + np.linalg.norm(logmap(_in[d*4:d*4+4, t+w-1:t+w],muMan))**2

        sigma = sigma/ws
        variation[0][t-1] = sigma
    mintab = np.array([])
    kp=1
    for t in range(variation.shape[1]):
        if kp==1:
            if variation[0][t]>thres:
                mintab = np.hstack((mintab,t+1))
                kp = 0
        else:
            if variation[0][t]<thres:
                mintab = np.hstack((mintab, t+1))
                kp=1
    cuts = np.array([mintab])
    return cuts, variation


def segmentSequence( _in, ws, thres ):  ##return 1-d vecteur and matrix
    variation = np.full([1, _in.shape[1]], np.nan)
    for t in range(math.ceil(ws / 2), _in.shape[1] - math.floor(ws / 2) + 1):
        sigma = 0
        for w in np.arange(-math.floor(ws/2),math.floor(ws/2)+1):
            for d in range(int(_in.shape[0]/4)):
                muMan = _in[d*4: d*4+4, t-1:t]
                sigma = sigma + np.linalg.norm(logmap(_in[d*4:d*4+4, t+w-1:t+w],muMan))**2

        sigma = sigma/ws
        variation[0][t-1] = sigma
    mintab = peakdet(variation, thres)[1]
    if mintab.shape[1]>0:
        cuts = mintab[:,0:1].T
    else:
        cuts = np.array([])
    return cuts, variation
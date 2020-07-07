#!/usr/bin/env python
import os
import json
import socket
import json
import time
from collections import OrderedDict
import logging
from logging.handlers import RotatingFileHandler
from threading import Thread
from functions.loadData import *
from Model import Model
from functions.basic_functions import *
from functions.estimateOrientationFromPosition import compute_q_from_dirbase

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

## Parameters
nbData = 300;  # Number of datapoints
nbSamples = 2;  # Number of demonstrations
trainName = 'data/Assis3Maxime/';  # folders names from where to load data
nspp = 2;  # number of skeleton sequence per folder
Model.nbVar = 46;  # Dimension of the tangent space (incl. time)
Model.nbVarMan = 55;  # Dimension of the manifold (incl. time)
nbIter = 10;  # Number of iteration for the Gauss Newton algorithm
nbIterEM = 10;  # Number of iteration for the EM algorithm
Model.nbStates = 15;  # Number of states in the GMM
Model.dt = 0.01;  # Time step duration
Model.params_diagRegFact = 1E-4;  # Regularization of covariance
registration = 1;  # temporal alignment or not
filt = 1;  # filtering of data or not
est = 1;  # estimation of orientation from position or kinect quaternions
rem = 1;  # removal of begining of the sequence (no motion) or not
ws = 21;  # windows size for segmentation
fastDP = 1;  # fast temporal alignment (using windows instead of full sequence) or not

x1 = np.array([1, 2, 3, 4])
x2 = np.array([6, 3, 1, 8])
x6=np.array([[1],[2],[3],[4]])
x5 = np.array([5,np.nan,4,2])
x3=np.vstack((x1,x2))
x10=np.vstack((x3,x3))
x7=x3[:,1:4]
iden = np.identity(4)
b = np.array([math.sqrt(b) for b in x1])
x1=np.array([x1])
fname = 'SkeletonSequence1.txt'

a = np.array([4,-2,1])
bt = np.array([1,-1,3])
c = compute_q_from_dirbase(a,bt)
x11 = np.array([[5,5],[5,5]])
x10[1:3,1:3] = x11
x4=np.array([[math.sqrt(b)+math.sqrt(b**2) for b in x6[0, :]],[b for b in x6[0, :]]])
x9=np.array([math.sqrt(b)+math.sqrt(b**2) for b in x6[0, :]])

test=np.array([[0 if np.isnan(b) else b for b in a] for a in x3])
# print(b)
# print(c)
# print(test)

# print(x10)
# print(np.dot(iden,x6))
# print(np.linalg.norm(x6))
# print(np.zeros(3))
# print(x3.shape)
# print(QuatMatrix(x6,axis=1))
# print(np.float64(1.0)/0.0)
# xt = np.linspace(0,10,21)
# yt=[np.sin(a) for a in xt]
# xx = np.linspace(0,10,41)
# print(xt)
# print(xx)
# print(yt)
# yy = np.array(splev(xx, splrep(xt, yt, k=3)))
# print(yy)

loadData(trainName,fname,filt, est, rem, ws, nbData)

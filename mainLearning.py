#!/usr/bin/env python
from threading import Thread
from Model import Model
from functions.processTrainingData import *
from functions.learnGMMmodel import learnGMMmodel
import pickle
import numpy as np

## Parameters
nbData = 300  # Number of datapoints
nbSamples = 2  # Number of demonstrations
trainName = 'data/Assis3Maxime/'  # folders names from where to load data
nspp = 2  # number of skeleton sequence per folder
nbVar = 46  # Dimension of the tangent space (incl. time)
nbVarMan = 55  # Dimension of the manifold (incl. time)
nbIter = 10  # Number of iteration for the Gauss Newton algorithm
nbIterEM = 10  # Number of iteration for the EM algorithm
nbStates = 15  # Number of states in the GMM
dt = 0.01  # Time step duration
params_diagRegFact = 1E-4  # Regularization of covariance
registration = 1  # temporal alignment or not
filt = 1  # filtering of data or not
est = 1  # estimation of orientation from position or kinect quaternions
rem = 1  # removal of begining of the sequence (no motion) or not
ws = 21  # windows size for segmentation
fastDP = 1  # fast temporal alignment (using windows instead of full sequence) or not

# x1 = np.array([0, -0.5, 3, 4])
#
# x2 = np.array([6, 3, 0, 8])
# x3 = np.array([9, 8, 4, 0])
# m1 = np.array([[6,9,-10, -1],[5,9,5,-5],[-8,7,-4,4],[-1,-9,0,6]])
# m2 = np.array([[1,2,3],[3,1,2],[2,3,1]])


model = Model(nbVar, nbVarMan, nbStates, dt, params_diagRegFact)
xIn, uIn, xOut, uOut = processTrainingData(model,trainName,nspp,registration,fastDP,filt,est,rem,ws,nbData)
u = uIn
x = xIn
for i in range(1,16):
    u = np.vstack((u, uOut[i]))
    x = np.vstack((x, xOut[i]))
model.x = x


# f = open('model1.txt', 'rb')
# model = pickle.load(f)
# u = pickle.load(f)
# xIn = pickle.load(f)
# xOut = pickle.load(f)
# f.close()


model = learnGMMmodel(model,u,xIn,xOut,nbSamples,nbIterEM,nbIter,nbData)


article = open('model.txt', 'wb')
pickle.dump(model, article)
article.close()

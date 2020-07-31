#!/usr/bin/env python
from functions.loadData import loadData
from functions.temporalAlignmentEval import temporalAlignmentEval
from functions.computeLikelihoods import computeLikelihoods
from functions.computeScores import computeScores
import time
import numpy as np
import pickle

## Parameters
nbData = 300  # Number of datapoints
seuil = -200  # Threshold used for computing score in percentage. The more close it is to zero, the more strict is the evaluation
minseuil = -500 # default values
registration = 1  # temporal alignment or not
filt = 1  # filtering of data or not
est = 1  # estimation of orientation from position or kinect quaternions
rem = 1  # removal of begining of the sequence (no motion) or not
ws = 21  # windows size for segmentation
fastDP = 1  # fast temporal alignment (using windows instead of full sequence) or not

# x1 = np.array([0, -0.5, 3, 4])
#
x2 = np.array([6, 3, -1, 8])
# x3 = np.array([9, 8, 4, 0])
# m1 = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, -9, 0, 6]])
m2 = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]])



# load modelExo3
f = open('model.txt', 'rb')
model = pickle.load(f)
f.close()

# data train for temporal alignment
dirTrain = 'data/Assis3Maxime/'
fnameTrain = 'SkeletonSequence1.txt'
oriMatTrain, posMatTrain, dataTrain = loadData(dirTrain, fnameTrain, filt, est, rem, ws, nbData)

# data test
dataTest = []
oriMatTestLong = []
posMatTestLong = []

dirTest = 'data/Assis1Maxime/'
fnameTest = 'SkeletonSequence3.txt'
[oriMatTest_, posMatTest_, dataTest_] = loadData(dirTest, fnameTest, filt, est, rem, ws, nbData)
dataTest.append(dataTest_)
oriMatTestLong.append(oriMatTest_)
posMatTestLong.append(posMatTest_)

## Evaluate sequence
for rep in range(len(dataTest)):
    if registration == 1:
        dataTestAligned, r, allPoses, poses, motion, distFI = temporalAlignmentEval(model, dataTrain,dataTest[rep],fastDP, nbData)
        posMatTest = posMatTestLong[rep][:,r]
    else:
        dataTestAligned = dataTest[rep]

    # compute likelihoods
    Lglobal, Lbodypart, Ljoints = computeLikelihoods(model, dataTestAligned)

    #get scores
    seuils = np.ones(6)*seuil
    minseuils = np.ones(6)*minseuil
    computeScores(model, Lglobal, Lbodypart, Ljoints, seuils, minseuils)


print(time.perf_counter())

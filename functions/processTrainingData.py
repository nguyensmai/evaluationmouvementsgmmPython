import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from functions.mapping import logmap
from functions.loadData import *
from functions.segmentSequence import *
from functions.temporalAlignment import *
import time


def processTrainingData(model, trainName, nspp, registration, fastDP, filt, est, rem, ws, nbData):
    uIn = np.array([])
    uOut = {}
    uRef = {}
    xOut = {}
    for i in range(1, 16):  # this dictionary starts with 1 and ends with 15
        uOut[i] = np.array([])

    for i in range(1, nspp+1):
        fname = 'SkeletonSequence' + str(i) + '.txt'
        dataTrain = loadData(trainName, fname, filt, est, rem, ws, nbData)[2]  ##1.5 second
        out = dataTrain['lElbow ori']
        out = np.vstack((out, dataTrain['lWrist ori']))
        out = np.vstack((out, dataTrain['lShoulder ori']))
        out = np.vstack((out, dataTrain['rElbow ori']))
        out = np.vstack((out, dataTrain['rWrist ori']))
        out = np.vstack((out, dataTrain['rShoulder ori']))
        cuts, variation = segmentSequence(out, ws, 0.05)  # optimized: 0.2s
        cutsKP = segmentSequenceKeyPose(out, ws, 0.02)[0]  # optimized: 0.2s
        if uIn.size == 0:
            model.cuts = cuts
            model.cutsKP = cutsKP
        if registration == 1:
            if uIn.size == 0:
                uRef = dataTrain
            else:
                dataTrain = temporalAlignment(uRef, dataTrain, fastDP)   ## optimisized :1s
        uIn = np.hstack((uIn, np.array(range(1, nbData + 1)) * model.dt))
        i = 1
        for d in dataTrain:
            if uOut[i].size == 0:
                uOut[i] = dataTrain[d]
            else:
                uOut[i] = np.hstack((uOut[i], dataTrain[d]))
            i += 1
    xIn = uIn
    std = np.array([[0], [1], [0], [0]])
    for i in range(1, 16):
        xOut[i] = uOut[i]
        if i < 10:
            uOut[i] = np.array([logmap(uOut[i][:, t:t + 1], std) for t in range(nbData * nspp)]).T[0]
        i += 1
    return xIn, uIn, xOut, uOut

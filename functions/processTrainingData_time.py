import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from functions.mapping import logmap
from functions.loadData import *
from functions.segmentSequence import *
from functions.temporalAlignment import *
import time


def processTrainingData(model, trainName, nspp, registration, fastDP, filt, est, rem, ws, nbData):
    t1 = time.perf_counter()
    uIn = np.array([])
    uOut = {}
    uRef = {}
    xOut = {}
    for i in range(1, 16):  # this dictionary starts with 1 and ends with 15
        uOut[i] = np.array([])

    for i in range(1, nspp + 1):
        fname = 'SkeletonSequence' + str(i) + '.txt'
        dataTrain = loadData(trainName, fname, filt, est, rem, ws, nbData)[2]  ##1.5 second
        t2 = time.perf_counter()
        print('t2-t1:',t2-t1)
        out = dataTrain['lElbow ori']
        out = np.vstack((out, dataTrain['lWrist ori']))
        out = np.vstack((out, dataTrain['lShoulder ori']))
        out = np.vstack((out, dataTrain['rElbow ori']))
        out = np.vstack((out, dataTrain['rWrist ori']))
        out = np.vstack((out, dataTrain['rShoulder ori']))

        cuts, variation = segmentSequence(out, ws, 0.05)  # 3 to 4 seconds to execute
        t3 = time.perf_counter()
        print('t3-t2:', t3 - t2)
        cutsKP = segmentSequenceKeyPose(out, ws, 0.02)[0]  # 3 to 4 seconds to execute
        t4 = time.perf_counter()
        print('t4-t3:', t4 - t3)
        if uIn.size == 0:
            model.cuts = cuts
            model.cutsKP = cutsKP
        if registration == 1:
            if uIn.size == 0:
                uRef = dataTrain
            else:
                dataTrain = temporalAlignment(uRef, dataTrain, fastDP)   ## takes more than 20 seconds to execute
        t5 = time.perf_counter()
        print('t5-t4:', t5 - t4)
        uIn = np.hstack((uIn, np.array(range(1, nbData + 1)) * model.dt))
        i = 1
        for d in dataTrain:
            if uOut[i].size == 0:
                uOut[i] = dataTrain[d]
            else:
                uOut[i] = np.hstack((uOut[i], dataTrain[d]))
            i += 1
    t6 = time.perf_counter()
    print('t6-t5:', t6 - t5)
    xIn = uIn
    std = np.array([[0], [1], [0], [0]])
    for i in range(1, 16):
        xOut[i] = uOut[i]
        if i < 10:
            uOut[i] = np.array([logmap(uOut[i][:, t:t + 1], std) for t in range(nbData * nspp)]).T[0]
        i += 1
    t7 = time.perf_counter()
    print('t7-t6:', t7 - t6)
    return xIn, uIn, xOut, uOut

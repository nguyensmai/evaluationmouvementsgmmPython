from functions.loadData import loadData
from functions.segmentSequence import *
from functions.temporalAlignment import temporalAlignment
from riepybdlib.manifold import get_euclidean_manifold, get_quaternion_manifold


def processTrainingData_pbdlib(model, trainName, nbSamples, registration, fastDP, filt, est, rem, ws, nbData):
    xIn = np.array([])
    uRef = {}
    xOut = {}

    for i in range(nbSamples):
        fname = 'SkeletonSequence' + str(i + 1) + '.txt'
        dataTrain = loadData(trainName, fname, filt, est, rem, ws, nbData)[2]  ##0.7 second
        out = dataTrain['lElbow ori']
        out = np.vstack((out, dataTrain['lWrist ori']))
        out = np.vstack((out, dataTrain['lShoulder ori']))
        out = np.vstack((out, dataTrain['rElbow ori']))
        out = np.vstack((out, dataTrain['rWrist ori']))
        out = np.vstack((out, dataTrain['rShoulder ori']))
        cuts, variation = segmentSequence(out, ws, 0.05)  # optimized: 0.2s
        cutsKP = segmentSequenceKeyPose(out, ws, 0.02)[0]  # optimized: 0.2s
        if i == 0:
            model.cuts = cuts
            model.cutsKP = cutsKP
        if registration == 1:
            if i == 0:
                uRef = dataTrain
            else:
                dataTrain = temporalAlignment(uRef, dataTrain, fastDP, nbData)[0]  ## optimisized :1s
        xIn = np.hstack((xIn, np.array(range(1, nbData + 1)) * model.dt))
        k = 1
        for d in dataTrain:
            if i == 0:
                xOut[k] = dataTrain[d]
            else:
                xOut[k] = np.hstack((xOut[k], dataTrain[d]))
            k += 1
    return xIn, xOut

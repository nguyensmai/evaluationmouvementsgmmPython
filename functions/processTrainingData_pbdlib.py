from functions.loadData import loadData
from functions.segmentSequence import *
from functions.temporalAlignment import temporalAlignment
from riepybdlib.manifold import get_euclidean_manifold, get_quaternion_manifold


def processTrainingData_pbdlib(model, trainName, nspp, registration, fastDP, filt, est, rem, ws, nbData):
    xIn = np.array([])
    uOut = {}
    uRef = {}
    xOut = {}
    for i in range(1, 16):  # this dictionary starts with 1 and ends with 15
        uOut[i] = np.array([])

    for i in range(nspp):
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
    uIn = xIn  ## x, u structure optimised
    std = np.array([[0], [1], [0], [0]])
    for k in range(1, 16):
        if k < 10:
            uOut[k] = logmap(xOut[k], std)
        else:
            uOut[k] = xOut[k]

    m_time = get_euclidean_manifold('time')  # actually it's 'in' in matlab
    m1 = get_quaternion_manifold('lElbow ori')
    m2 = get_quaternion_manifold('lWrist ori')
    m3 = get_quaternion_manifold('lShoulder ori')
    m4 = get_quaternion_manifold('rElbow ori')
    m5 = get_quaternion_manifold('rWrist ori')
    m6 = get_quaternion_manifold('rShoulder ori')
    m7 = get_quaternion_manifold('mSpine ori')
    m8 = get_quaternion_manifold('mShoulder ori')
    m9 = get_quaternion_manifold('Neck ori')
    m10 = get_euclidean_manifold(3, 'lElbow rel_pos')
    m11 = get_euclidean_manifold(3, 'lWrist rel_pos')
    m12 = get_euclidean_manifold(3, 'lShoulder rel_pos')
    m13 = get_euclidean_manifold(3, 'rElbow rel_pos')
    m14 = get_euclidean_manifold(3, 'rWrist rel_pos')
    m15 = get_euclidean_manifold(3, 'rShoulder rel_pos')

    manifold = m_time * m1 * m2 * m3 * m4 * m5 * m6 * m7 * m8 * m9 * m10 * m11 * m12 * m13 * m14 * m15

    return xIn, uIn, xOut, uOut

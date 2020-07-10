import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from functions.loadData import *
from functions.segmentSequence import *

def processTrainingData(model, trainName, nspp,registration,fastDP,filt,est,rem,ws,nbData ):
    uIn = np.array([])
    uOut={}
    for i in range(1,16):  #this dictionary starts with 1 and ends with 15
        uOut[i] = np.array([])

    for i in range(1,nspp+1):
        fname = 'SkeletonSequence'+str(i)+'.txt'
        dataTrain = loadData(trainName,fname,filt,est,rem,ws,nbData)[2]
        out = dataTrain['lElbow ori']
        out = np.vstack((out, dataTrain['lWrist ori']))
        out = np.vstack((out, dataTrain['lShoulder ori']))
        out = np.vstack((out, dataTrain['rElbow ori']))
        out = np.vstack((out, dataTrain['rWrist ori']))
        out = np.vstack((out, dataTrain['rShoulder ori']))

        cuts, variation = segmentSequence(out, ws, 0.05)
        print(dataTrain)


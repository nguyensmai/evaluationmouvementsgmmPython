import numpy as np
from functions.mapping import logmap
from functions.dp import *
import time

def temporalAlignment(trainD, testD, fast):
    # Dynamic programming to align two sequences train and test
    # fast may be set to 1 to speed process by using small windows of 70
    # Compute distances
    size = testD['lElbow ori'].shape[1]
    DM = np.zeros((size, size))
    for row in range(0, size):
        DMline = logmap(testD['lElbow ori'], trainD['lElbow ori'][:, row:row + 1])
        DMline = np.vstack((DMline,logmap(testD['lWrist ori'], trainD['lWrist ori'][:, row:row + 1])))
        DMline = np.vstack((DMline, logmap(testD['lShoulder ori'], trainD['lShoulder ori'][:, row:row + 1])))
        DMline = np.vstack((DMline, logmap(testD['rElbow ori'], trainD['rElbow ori'][:, row:row + 1])))
        DMline = np.vstack((DMline, logmap(testD['rWrist ori'], trainD['rWrist ori'][:, row:row + 1])))
        DMline = np.vstack((DMline, logmap(testD['rShoulder ori'], trainD['rShoulder ori'][:, row:row + 1])))
        DM[row,:] = np.linalg.norm(DMline,axis=0)
    if fast == 1:
        for row in range(size-70):
            for col in range(70+row,size):
                DM[row][col] = 5
        for col in range(size-70):
            for row in range(70+col,size):
                DM[row][col] = 5

    p, q = dp(DM)  ## execution time : 1s
    # alignment
    r = np.zeros(size)
    for t in range(size):
        ind = np.where(p == t)[0]
        if ind.size == 0:
            r[t] = p[t]
        else:
            r[t] = q[ind[0]]
    # out
    r = r.astype(int)
    out = {}    ## TODO : maybe we can avoid making another dictionary
    for key in testD:
        out[key] = testD[key][:, r]
    return out

import numpy as np
from functions.logmap import logmap
from functions.dp import *


def temporalAlignment(trainD, testD, fast):
    # Dynamic programming to align two sequences train and test
    # fast may be set to 1 to speed process by using small windows of 70
    # Compute distances
    size = testD['lElbow ori'].shape[1]
    DM = np.ones((size, size)) * 5
    for row in range(0, size):
        for col in range(0, size):
            if fast == 1:
                if np.abs(row - col) < 70:
                    DM[row][col] = np.linalg.norm(
                        [logmap(testD['lElbow ori'][:, col:col + 1], trainD['lElbow ori'][:, row:row + 1]),
                         logmap(testD['lWrist ori'][:, col:col + 1], trainD['lWrist ori'][:, row:row + 1]),
                         logmap(testD['lShoulder ori'][:, col:col + 1], trainD['lShoulder ori'][:, row:row + 1]),
                         logmap(testD['rElbow ori'][:, col:col + 1], trainD['rElbow ori'][:, row:row + 1]),
                         logmap(testD['rWrist ori'][:, col:col + 1], trainD['rWrist ori'][:, row:row + 1]),
                         logmap(testD['rShoulder ori'][:, col:col + 1], trainD['rShoulder ori'][:, row:row + 1])])
            else:
                DM[row][col] = np.linalg.norm(
                    [logmap(testD['lElbow ori'][:, col:col + 1], trainD['lElbow ori'][:, row:row + 1]),
                     logmap(testD['lWrist ori'][:, col:col + 1], trainD['lWrist ori'][:, row:row + 1]),
                     logmap(testD['lShoulder ori'][:, col:col + 1], trainD['lShoulder ori'][:, row:row + 1]),
                     logmap(testD['rElbow ori'][:, col:col + 1], trainD['rElbow ori'][:, row:row + 1]),
                     logmap(testD['rWrist ori'][:, col:col + 1], trainD['rWrist ori'][:, row:row + 1]),
                     logmap(testD['rShoulder ori'][:, col:col + 1], trainD['rShoulder ori'][:, row:row + 1])])
    p, q = dp(DM)

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
    out = {}
    for key in testD:
        out[key] = testD[key][:, r]
    return out

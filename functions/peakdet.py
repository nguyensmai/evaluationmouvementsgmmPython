import math
import numpy as np


def peakdet(v, delta, *x):
    #
    maxtab = np.array([])
    mintab = np.array([])
    print(maxtab.size)

    v = v.T  ##v was a horizontal vector
    print(v)
    if x:
        print(x[0])
        x = x[0].T
        if v.shape[0] != x.shape[0]:
            pass  # error
    else:
        print(v.shape[0])
        x = np.arange(1, v.shape[0] + 1).reshape([v.shape[0], 1])

    ##delta must be a positive scalar

    mn=np.inf
    mx=-np.inf
    mnpos = np.nan
    mxpos = np.nan
    lookformax = 1

    for i in range(v.shape[0]):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax!=0:
            if this < mx-delta:
                if maxtab.size!=0:
                    maxtab = np.vstack((maxtab,[mxpos, mx]))
                else:
                    maxtab = np.array([mxpos, mx])
                mn = this
                mnpos = x[i]
                lookformax = 0
        else:
            if this > mn+delta:
                if mintab.size != 0:
                    mintab = np.vstack((mintab, [mnpos, mn]))
                else:
                    mintab = np.array([mnpos, mn])
                    mx = this
                    mxpos = x[i]
                    lookformax = 1
    return maxtab, mintab




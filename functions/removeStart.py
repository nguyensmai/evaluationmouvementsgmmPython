import math
import numpy as np
from functions.mapping import logmap
from riepybdlib.mappings import quat_log_e

def removeStart(_in, ws, thres, dec):
    for t in range(math.ceil(ws / 2), _in.shape[1] - math.floor(ws / 2) + 1):
        sigma = 0
        for w in np.arange(-math.floor(ws/2),math.floor(ws/2)+1):  ## suppose that ws is always integer
            for d in range(int(_in.shape[0]/4)):
                muMan = _in[d*4: d*4+4, t-1:t]
                sigma = sigma + np.linalg.norm(logmap(_in[d*4:d*4+4, t+w-1:t+w],muMan))**2

        sigma = sigma/ws
        if sigma>thres:
            deb = max(1,t-dec)
            out = _in[:,deb-1:]
            return out, deb
    deb = 1
    out = _in
    return out, deb

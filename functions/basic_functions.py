import math
import numpy as np

def acoslog(x):
    '''
    Arcosine re-defitinion to make sure the distance between antipodal quaternions is zero (2.50 from Dubbelman's Thesis)
    '''
    acosx = 0
    if x >= 1.0:
        x = 1.0
    if x <= -1.0:
        x = -1.0
    if x>=-1.0 and x<0:
        acosx = math.acos(x)-math.pi
    else:
        acosx = math.acos(x)
    return acosx

def QuatMatrix(q,axis=0): ##0 for horizontal vector, 1 for vertical vector
    if axis==1:
        return np.array([[q[0][0], -q[1][0], -q[2][0], -q[3][0]],
                         [q[1][0], q[0][0], -q[3][0], q[2][0]],
                         [q[2][0], q[3][0], q[0][0], -q[1][0]],
                         [q[3][0], -q[2][0], q[1][0], q[0][0]]
                         ])
    else:
        return np.array([[q[0], -q[1], -q[2], -q[3]],
                     [q[1], q[0], -q[3], q[2]],
                     [q[2], q[3], q[0], -q[1]],
                     [q[3], -q[2], q[1], q[0]]])


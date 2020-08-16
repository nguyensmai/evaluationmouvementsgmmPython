'''
Writing code takes time. Polishing it and making it available to others takes longer! 
If some parts of the code were useful for your research of for a better understanding 
of the algorithms, please reward the authors by citing the related publications, 
and consider making your own research available in this way.

@article{Zeestraten2017,
  	title = {An Approach for Imitation Learning on Riemannian Manifolds},
	author = {Zeestraten, M.J.A. and Havoutis, I. and Silverio, J. and Calinon, S. and Caldwell, D. G.},
	journal={{IEEE} Robotics and Automation Letters ({RA-L})},
	year = {2017},
	month={January},
}

 
Copyright (c) 2017 Istituto Italiano di Tecnologia, http://iit.it/
Written by Martijn Zeestraten, http://www.martijnzeestraten.nl/

This file is part of RiePybDlib, http://gitlab.martijnzeestraten.nl/martijn/riepybdlib

RiePybDlib is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.
 
RiePybDlib is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RiePybDlib. If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np


########################### EULER ANGLES #############################################################
# Define basic rotations:
# We use the conventions r = R*r'
# i.e. a unit rotation defines a rotation from the body fixed coordinates to the world-frame coordinates


def fRx(theta):

    return np.array([[1, 0       , 0           ],
                      [0, np.cos(theta),-np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]
                     ])
def fRy(theta):
    return np.array([[np.cos(theta), 0 , np.sin(theta)],
                      [0            , 1 , 0 ],
                      [-np.sin(theta), 0 , np.cos(theta)]
                     ])
    
def fRz(theta):
    return np.array([[np.cos(theta), -np.sin(theta)  , 0],
                      [np.sin(theta), np.cos(theta) , 0],
                      [0            , 0           , 1]
                    ])


################################### QUATERNIONS 


def skew(q):
    return np.array([ [ 0   ,-q[2], q[1] ], 
                      [ q[2], 0   ,-q[0] ],
                      [-q[1], q[0], 0    ] ])

class Quaternion(object):
    def __init__(self,q0,q):
        '''
        a quaterion consist of a scalar q0 and a three dimensional vector q
        '''
        if (len(q) == 3):
            self.q0 = float(q0) 
            self.q  = q 
        else:
            print('.q.shape {0}'.format(q.shape))
            raise TypeError
        
    def __neg__(self):
        return Quaternion(-self.q0,-self.q)
    
    def __add__(self,other):
        '''
        q = self
        p = other
        q + p = (q0 + p0, qvec + pvec)
        '''

        if type(other) is Quaternion:
            return Quaternion(self.q0 + other.q0,self.q+other.q) 
        elif type(other) is list:
            qlist = []
            for _,q in enumerate(other):
                qlist.append(self+q)
            return qlist
    
    def __sub__(self,other):
        '''
        q = self
        p = other
        q - p = (q0 - p0, qvec - pvec)
        '''
        if type(other) is Quaternion:
            return Quaternion(self.q0 - other.q0, self.q - other.q)
        elif type(other) is list:
            qlist = []
            for _,q in enumerate(other):
                qlist.append(self-q)
            return qlist

        
        
    def __mul__(self,other):
        ''' quaternion product (non-commutative):
            q = self
            p = other
            'x' indicates the cross product
            q*p = (q0*p0 - qvec*pvec, q0*pvec + p0*qvec + qvec x pvec)
        
        '''

        if type(other) is Quaternion:
            v0 = self.q0*other.q0 - self.q.dot(other.q)
            v  = self.q0*other.q + other.q0*self.q + np.cross(self.q,other.q)
            return Quaternion(v0,v)
        elif type(other) is list:
            qlist = []
            for _,q in enumerate(other):
                qlist.append(self*q)
            return qlist

    
    def adj(self):
        '''
        The adjoint of the quaternion (q0, -qvec)
        '''
        return Quaternion(self.q0,-self.q)
    
    def norm(self):
        #return np.sqrt( (self.adjoint() * self).q0 )
        return np.sqrt(self.q0**2 + self.q.dot(self.q))
        
    def Q(self):
        '''
        Quaternion matrix
        '''
        Q1   = np.hstack(( self.q0, -self.q ) )
        Q234 = np.hstack( (self.q[:,None] , self.q0*np.eye(3) + skew(self.q) )) 
        Q = np.vstack((Q1,Q234))
        
        return Q

    # @staticmethod
    # def to_nparray_st(data):
    #     if type(data) is Quaternion:
    #         return data.to_nparray()
    #     else:
    #         qarray = np.zeros( (len(data), 4) )
    #         for i,q in enumerate(data):
    #             if type(q) is Quaternion:
    #                 qarray[i,:] = q.to_nparray()
    #             elif type(q) is np.ndarray:
    #                 qarray[i, :] = q
    #             else:
    #                 raise RuntimeError('Could not determine dimension of input')
    #         return qarray

    @staticmethod
    def to_nparray_st(data):
        if type(data) is Quaternion:
            return data.to_nparray()
        else:
            qarray = np.zeros( (len(data), 4) )
            for i,q in enumerate(data):
                qarray[i,:] = q.to_nparray()
            return qarray

    def to_nparray(self):
            return np.hstack( ([self.q0],self.q))

    @staticmethod
    def from_R(R):
        '''Return (list of) Quaternion from (list of) Rotation Matrix'''
        if type(R) is list:
            res = []
            for Rtmp in R:
                res.append(Quaternion.from_R(Rtmp))
            return res
        else:
            return get_q_from_R(R)

    @staticmethod
    def from_nparray(qarray):
        ''' Return list of Quaternions from an np array
        qarray: n_data x 4 numpy array in which each column is [q0, q1, q2, q3]
        '''
        if qarray.ndim==1:
            # Single sample:
            return Quaternion(qarray[0], qarray[1:])
        else:
            qlist = []
            for i in range(qarray.shape[0]):
                qlist.append(Quaternion(qarray[i,0], qarray[i,1:]))
            return qlist


    def adjQ(self):
        '''
        Quaternion matrix
        '''
        Q1   = np.hstack(( self.q0, -self.q ) )
        Q234 = np.hstack( (self.q[:,None] , self.q0*np.eye(3) - skew(self.q) )) 
        Q = np.vstack((Q1,Q234))
        
        return Q
    
    
    def normalized(self):
        norm = self.norm()
        return Quaternion(self.q0/norm, self.q/norm)
    
    
    def i(self):
        ''' Reciprocal (inverse) of a Quaternion'''
        qbar = self.adj()
        norm2 = self.norm()**2
        return Quaternion(qbar.q0/norm2, qbar.q/norm2)
    
    
    def R(self):
        ''' From Peter Corke's Matlab robotics toolbox'''
        s = self.q0
        x = self.q[0]
        y = self.q[1]
        z = self.q[2]

        R = np.array([
            [1-2*(y**2+z**2), 2*(x*y-s*z)    , 2*(x*z+s*y)],
            [2*(x*y+s*z)    , 1-2*(x**2+z**2), 2*(y*z-s*x)],
            [2*(x*z-s*y)    , 2*(y*z+s*x)    , 1-2*(x**2+y**2)]
            ])
        return R
    
    def __str__(self):
        return "({0:.2f}, {1})".format(self.q0,self.q) 

############## Rotational Conversions:

def quatToEulerXYZ(q):
    q = q.normalized()
    R = q.R();
    
    # NASA paper:
    th1 = np.arctan2(-R[1,2],R[2,2])
    th2 = np.arctan2(R[0,2],np.sqrt(1-R[0,2]**2))
    th3 = np.arctan2(-R[0,1],R[0,0])
    
    
    return np.array([th1,th2,th3])


def get_q_from_R(R):
    ''' From Peter Corke's Robotics toolbox'''

    qs = min(np.sqrt( np.trace(R) +1)/2.0,1.0)
    kx = R[2,1] - R[1,2]   # Oz - Ay
    ky = R[0,2] - R[2,0]   # Ax - Nz
    kz = R[1,0] - R[0,1]   # Ny - Ox

    if (R[0,0] >= R[1,1]) and (R[0,0] >= R[2,2]) :
        kx1 = R[0,0] - R[1,1] - R[2,2] + 1 # Nx - Oy - Az + 1
        ky1 = R[1,0] + R[0,1]              # Ny + Ox
        kz1 = R[2,0] + R[0,2]              # Nz + Ax
        add = (kx >= 0)
    elif (R[1,1] >= R[2,2]):
        kx1 = R[1,0] + R[0,1]              # Ny + Ox
        ky1 = R[1,1] - R[0,0] - R[2,2] + 1 # Oy - Nx - Az + 1
        kz1 = R[2,1] + R[1,2]              # Oz + Ay
        add = (ky >= 0)
    else:
        kx1 = R[2,0] + R[0,2]              # Nz + Ax
        ky1 = R[2,1] + R[1,2]              # Oz + Ay
        kz1 = R[2,2] - R[0,0] - R[1,1] + 1 # Az - Nx - Oy + 1
        add = (kz >= 0)

    if add:
        kx = kx + kx1
        ky = ky + ky1
        kz = kz + kz1
    else:
        kx = kx - kx1
        ky = ky - ky1
        kz = kz - kz1

    nm = np.linalg.norm(np.array([kx, ky, kz]))
    if nm == 0:
        q = Quaternion(1, np.zeros(3))
    else:
        s = np.sqrt(1 - qs**2) / nm
        qv = s*np.array([kx, ky, kz])
        q = Quaternion(qs, qv)

    return q


#### Axis-Angle Representation:
def get_axisangle(d, e=None, reg=1e-6):
    ''' Compute axis angle of axis de w.r.t e
        d: Vector [1d ndarray]
        e: Vector [1d ndarray]
        
        Computation:
        ax    = (e x d) / (|| e x d ||)
        angle = arccos( d * e ) 

        * is in product
        x is the cross product

        if no e is provided, we adopt: e = [0, 0, 1]^T

    '''

    if e is None:
        e = np.array([0,0,1])
        norm = np.sqrt(d[0]**2 + d[1]**2)
        if norm < reg:
            return (e,0)
        else:
            vec = np.array( [-d[1], d[0], 0])
            return vec/norm, np.arccos(d[2])
    else:
        # Compute cross product between identity and d
        exd = skew(e).dot(d)
        norm = np.linalg.norm(exd)

        # Check norm:        
        if norm < reg:
            # smaller than reguralization, assume no rotation:
            ax = e
            angle = 0
        else:
            # Rotation is present:
            ax = exd/norm
            angle = np.arccos( (d*e).sum( axis=(d.ndim-1) ) )
        return (ax, angle)


def R_from_axis_angle(ax, angle):
    ''' Get Rotation matrix from axis angle representation using Rodriguez formula
        ax   : The unit axis defining the axis of rotation [ 1d ndarray]
        angle: Angle of rotation [float]

        Return: 
        R(ax, angle) = I + sin(angle) x ax + (1 - cos(angle) ) x ax^2

        where x is the cross product
    '''

    utilde = skew(ax)
    return np.eye(3) + np.sin(angle)*utilde + (1 - np.cos(angle))*utilde.dot(utilde)

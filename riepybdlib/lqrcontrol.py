'''
Riemannian statistics module of riepybdlib package

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



import abc
ABC = abc.ABCMeta('ABC', (object,), {})

import numpy as np
import scipy as sp
import scipy.linalg as splinalg


import threading as thrd
import time as time

import riepybdlib.manifold as rm
import riepybdlib.statistics as rs


class ControllerThread(object):
    """ Thread that operates provided function floop at fixed rate 
    """
    def __init__(self, threadrate, floop):
        """Initialize """
        self._dt = 1.0/threadrate
        self._floop = floop
        self.__running = False

    def __worker(self):
        """Thread loop"""

        t_start = time.time()
        while (self.__running==True):
            # Current state:
            self._floop()

            # Control loop speed:
            elapsed = (time.time() - t_start)
            tsleep = max(self._dt - (elapsed), 0.0)
            time.sleep(tsleep)
            t_start += (elapsed+tsleep)

    def start(self):
        """Start thread"""
        # Start the system
        self.__running = True
        self.__thread = thrd.Thread(target=self.__worker).start()

    def stop(self):
        """Stop thread"""
        self.__running = False

    @property
    def isrunning(self):
        """Inidication of running thread"""
        return self.__running

def dlqr(A,B,Q,R):
    ''' Returns gain matrix after solving discrete infinite horizine lqr.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    '''
    
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = np.matrix(splinalg.solve_discrete_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.array(sp.linalg.inv(B.T*X*B+R)*(B.T*X*A))
     
    return K;

def create_discretelinearsystem(invM, C, dt):
    '''Construct the canonical system A*x + B*u'''
    n_vars = invM.shape[0] 
    A = np.zeros( (n_vars*2, n_vars*2) )
    A[:n_vars, n_vars:] = np.eye(n_vars)
    A[n_vars:, n_vars:] = invM.dot(-C)
    A = A*dt + np.eye(n_vars*2)
    B = np.zeros( (n_vars*2, n_vars) )
    B[n_vars:,:] = invM*dt

    return (A,B)

def get_controlmanifold(manifold):
    ''' Function returns argument manifold augmented with Euclidean velocity manifolds'''
    # Define control manifold:
    for i in range(manifold.n_manifolds):
        subman = manifold.get_submanifold(i)
        if i==0:
            velman = rm.get_euclidean_manifold(subman.n_dimT,
                    '{1}D TS of {0}'.format(subman.name, subman.n_dimT ))
        else:
            velman *= rm.get_euclidean_manifold(subman.n_dimT,
                    '{1}D TS of {0}'.format(subman.name, subman.n_dimT ))
    return manifold*velman



class AbstractRiemannianLQR(ABC):
    def __init__(self,manifold, A, B, Q, R, controlrate=500):
        self.manifold = get_controlmanifold(manifold)

        # Assign all properties:
        self.__initializing=True
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.setpoint = self.manifold.id_elem
        self.__controlcommand = np.zeros(manifold.n_dimT)
        self.__initializing=False

        # Compute gains of Riemannian LQR
        self._computeLQR()

        # Create controller loop
        self._controller = ControllerThread(controlrate, self._lqrstep)

    def start(self):
        '''Start LQR controller'''
        self.setpoint = self.getstate()
        self._controller.start()

    def stop(self):
        '''Stop LQR controller'''
        self._controller.stop()

    def _computeLQR(self):
        '''Compute the LQR gains'''
        if not self.__initializing:
            self.LQRgains = dlqr(self.A, self.B, self.Q, self.R)

    def _lqrstep(self):
        '''Function that is executed each control loop'''
        cur_state = self.getstate()
        err = self.manifold.log(cur_state, self.setpoint)
        tmp = -self.LQRgains.dot(err)
        self.controlcommand = tmp

    @abc.abstractmethod
    def getstate(self):
        pass

    # Properties:
    @property 
    def controlcommand(self):
        '''Control command'''
        return self.__controlcommand

    @controlcommand.setter
    def controlcommand(self, val):
        '''Control command'''
        self.__controlcommand = val
        self.controlcommandsetter(val)
        #self.__controlcommand=val

    @abc.abstractmethod
    def controlcommandsetter(self, val):
        pass

    @property
    def Q(self):
        '''Tracking cost matrix'''
        return self.__Q

    @Q.setter
    def Q(self, val):
        '''Tracking cost matrix'''
        
        self.__Q = val
        self._computeLQR()

    @property
    def R(self):
        '''Control cost matrix'''
        return self.__R

    @R.setter
    def R(self, val):
        '''Control cost matrix'''
        self.__R = val
        self._computeLQR()

    @property
    def A(self):
        '''Linear System dynamics'''
        return self.__A

    @A.setter
    def A(self, val):
        '''Linear System dynamics'''
        self.__A = val
        self._computeLQR()

    @property
    def B(self):
        '''Input Dynamics'''
        return self.__B

    @B.setter
    def B(self, val):
        '''Input Dynamics'''
        self.__B = val
        self._computeLQR()

    @property
    def setpoint(self):
        '''LQR setpoint'''
        return self.__setpoint

    @setpoint.setter
    def setpoint(self, val):
        '''LQR setpoint'''
        self.__setpoint = val

    @property
    def LQRgains(self):
        '''LQR gains'''
        return self.__LQRgains

    @LQRgains.setter
    def LQRgains(self, val):
        '''LQR gains'''
        self.__LQRgains = val
    

    


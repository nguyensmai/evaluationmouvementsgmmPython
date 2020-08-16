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

import numpy as np
import scipy as sp
import scipy.linalg 
import riepybdlib.angular_representations as ar

# Some add functions to transform matrices into vectors and back
def matrix2vec(S):
    ''' Conversion of Matrix to equivalent Vector representation'''
    if type(S) is list:
        res = []
        for tmpS in S:
            res.append(symmetricmatrix2vec(S[i,:]))
        return np.vstack(res)
    else:
        (d,_) = S.shape
        return S.reshape(-1)
    
def vec2matrix(v):
    ''' Conversion a vector to a matrix.'''
    if v.ndim==2:
        res = []
        res = []
        for i in range(S.shape[0]):
            res.append( matrix2vec(S[i,:]))
        return res
    else:
        t = v.shape[-1];
        d = int(np.sqrt(v.shape[-1]))
        return v.reshape(d,d)



# ---------------- Manifold mapping functions
# Euclean space Mappings:
def eucl_action(x,g,h):
    return x - g + h

def eucl_exp_e(g_tan,reg=None):
    return g_tan

def eucl_log_e(g,reg=None):
    return g

def eucl_parallel_transport(Xg, g, h, t=1):
    return Xg


# -----------------     Quaternion Action, Log and Exponential        
def arccos_star(rho):
    if type(rho) is not np.ndarray:
        # Check rho
        if abs(rho)>1:
            # Check error:
            if (abs(rho) - 1 > 1e-6):
                print('arcos_star: abs(rho) > 1+1e-6:'.format( abs(rho)-1) )
                
            # Fix error
            rho = 1*np.sign(rho)
        
        # Single mode:
        if (-1.0 <= rho and rho < 0.0):
            return np.arccos(rho) - np.pi
        else:
            return np.arccos(rho)
    else:
        # Batch mode:
        rho = np.array([rho])
        
        ones = np.ones(rho.shape)
        rho = np.max(np.vstack( (rho,-1*ones ) ), axis=0)
        rho = np.min(np.vstack( (rho, 1*ones ) ), axis=0)

        acos_rho = np.zeros(rho.shape)
        sl1 = np.ix_ ((-1.0 <= rho)*(rho < 0.0)==1)
        sl2 = np.ix_ ((1.0 > rho)*(rho >= 0.0)==1)

        acos_rho[sl1] = np.arccos(rho[sl1]) - np.pi
        acos_rho[sl2] = np.arccos(rho[sl2])

        return acos_rho

quat_id = ar.Quaternion(1, np.zeros(3) )

def quat_action(x,g,h):
    return h*g.i()*x

def quat_log_e(g, reg=1e-6):
    d_added = False
    if type(g) is list:
        # Batch mode:
        g_np = ar.Quaternion.to_nparray_st(g)
    
        # Create tangent values, and initalize to zero
        g_tan = np.zeros( (g_np.shape[0], 3) )

        # compute tangent values for quaternions which have q0 other than 1.0 - reg
        # Slices:
        g_np0_abs = abs(g_np[:,0]-1.0) > reg
        sl_123 = np.ix_(g_np0_abs, range(1,4) )
        sl_012 = np.ix_(g_np0_abs, range(3) )
        sl_0   = np.ix_(g_np0_abs, [0] )

        # Compute tangent values:
        acos_q0 = arccos_star(g_np[sl_0][:,0])
        qnorm   = g_np[sl_123].T/ np.linalg.norm(g_np[sl_123], axis=1)
        g_tan[sl_012] = (qnorm*acos_q0).T

        return g_tan
    else:
        # Single mode:
        if abs(g.q0 - 1.0)>reg:
            return arccos_star(g.q0)* (g.q/np.linalg.norm(g.q))
        else:
            return np.zeros(3)
    
    
# The exponential map:
def quat_exp_e(g_tan, reg=1e-6):
    if g_tan.ndim == 2:
        # Batch mode:
        qnorm = np.linalg.norm(g_tan, axis=1)
        qvec = np.vstack( (np.ones( g_tan.shape[0]), np.zeros( g_tan.shape ).T) ).T

        # Select the non identity quaternions:
        qnorm_abs = abs(qnorm) > reg
        sl_0123 = np.ix_( qnorm_abs, range(4) )
        sl_012 =  np.ix_( qnorm_abs, range(3) )
        sl     =  np.ix_( qnorm_abs)
        qnorm = qnorm[sl] 

        # Compute the non identity values:
        qvec[sl_0123]  = np.vstack( (np.cos(qnorm), g_tan[sl_012].T*(np.sin(qnorm)/qnorm) ) ).T

        # Generate resulting quaternions:
        res = []
        for i in range(g_tan.shape[0]):
            res.append(ar.Quaternion(qvec[i,0], qvec[i,1:]) )  # Return result:
        #if d_added:
        #    return res[0] # return on the single quaternion that was requested
        #else:
        return res # Return the list of quaternions
    else:
        # Single mode:
        qnorm = np.linalg.norm(g_tan)
        if ( qnorm != 0 ):
            return ar.Quaternion( np.cos(qnorm), np.sin(qnorm)*( g_tan/qnorm )) 
        else:
            return ar.Quaternion(1, np.zeros(3) )

def quat_log(x,g, reg=1e-6):
    return quat_log_e(g.i()*x, reg)

def quat_exp(x,g, reg=1e-6):
    return g*quat_exp_e(x, reg)

def quat_parallel_transport(Xg, g, h, t=1):
    ''' Parallel transport of vectors in X from h to g*t, 0 <= t <= 1
        Implementation is modified version of the one reported in
        Optimization algorithms on Manifolds:
    '''
    
    # Get intermediate position on geodesic between g and h 
    if t<1:
        ht = quat_exp( quat_log(h, g)*t, g)
    else:
        ht=h
    
    # Get quaternion matrices for rotation computation
    Qeg = g.Q()  # Rotation between origin and g
    Qeh = ht.Q() # between  rotation between ht and origin
    
    
    # Get tangent  vector of h in g, expressed in R^4
    # We first construct it at the origin (by augmenting it with 0)
    # and then rotate it to point g
    v = Qeg.dot(np.hstack([[0], quat_log(h,g)]))
        
    # Transform g into np array for computations
    gnp = g.to_nparray()  # Transform to np array
    m = np.linalg.norm(v) # Angle of rotation (or 'transport)
    
    # Compute Tangential rotation (this is done in n_dimM)
    if m < 1e-6:
        Rgh = np.eye(4)
    else:
        u = (v/m)[:,None]
        gnp = gnp[:,None]

        Rgh = (- gnp*np.sin( m*t )*u.T 
             + u*np.cos( m*t )*u.T
             + ( np.eye(4) - u.dot(u.T) )
            )
        
    # ----- Finally compute rotation compensation to achieve parallel transport:
    Ie = np.eye(4)[:,1:].T
    
    Ig   = Ie.dot(Qeg.T)                   # Base orientation at g
    Ih   = Ig.dot(Rgh.T)                   # Base orientation at h by parallel transport
    Ie_p = Ih.dot(Qeh)                     # Tangent space orientation at origin with parallel transport
    
    # Compute relative rotation:
    R = Ie.dot(Ie_p.T)
    
    if np.sign(np.trace(R)) ==-1:
        # Change sign, to ensure proper rotation
        R = -R
                   
    # Transform points and return
    return Xg.dot(R.T)    
        
# ----------------------  S^2,
s2_id = np.array([0,0,1])
def s2_action(x, g, h):
    ''' Moves x relative to g, to y relative to h

    '''
    # Convert possible list into nparray
    if type(x) is list:
        x = np.vstack(x)

    # Get rotation of origin e to g
    ax, angle = ar.get_axisangle(g)
    Reg = ar.R_from_axis_angle(ax, angle)

    # Get rotation of origin e to h
    ax, angle = ar.get_axisangle(h)
    Reh = ar.R_from_axis_angle(ax, angle)

    # Creat rotation that moves x from g to the origin e,
    # and then from e to h:
    A = Reh.dot(Reg.T)

    return x.dot(A.T)

def s2_exp_e(g_tan, reg=1e-6):
    if g_tan.ndim ==2:
        # Batch operation:
    
        # Standard we assume unit values:
        val = np.vstack( (
                          np.zeros( (2, g_tan.shape[0]) ), 
                          np.ones( g_tan.shape[0] ) 
                         ) 
                       ).T
        
        # Compute distance for all values that are larger than the regularization:
        norm = np.linalg.norm(g_tan,axis=1)
        cond = norm> reg
        sl_2 =  np.ix_( cond, [2] )
        sl_01 = np.ix_( cond, range(0,2) )

        norm = norm[np.ix_(cond)]         # Throw away the norms we don't use
        gt_norm = (g_tan[sl_01].T/norm).T # Normalize the input vector of selected values

        val[sl_2]  = np.cos(norm)[:,None]
        val[sl_01] = (gt_norm.T*np.sin(norm)).T

        return val 
    else:
        # Single mode:
        norm = np.linalg.norm(g_tan)
        if norm > reg:
            gt_norm = g_tan/norm
            return np.hstack( (np.sin(norm)*gt_norm, [np.cos(norm)]) )
        else:
            return np.array([0,0,1])
    
def s2_log_e(g, reg = 1e-10):
    '''Map values g, that lie on the Manifold, into the tangent space at the origin'''
    
    # Check input
    d_added = False
    if g.ndim ==2:
        # Batch operation,
        # Assume all values lie at the origin:
        val = np.zeros( (g.shape[0], 2) )

        # Compute distance for all values that are larger than the regularization:
        cond =  (1-g[:,2]) > reg
        sl_2 =  np.ix_( cond, [2] )
        sl_01 = np.ix_( cond, range(0,2) )

        val[sl_01] = (g[sl_01].T*( np.arccos( g[sl_2])[:,0]/np.linalg.norm(g[sl_01], axis=1) ) ).T
        return val 
    else:
        # single mode:
        if abs(1-g[2]) > reg:
            return np.arccos(g[2])*g[0:2]/np.linalg.norm(g[0:2])
        else:
            return np.array([0,0])

def s2_exp(x, g, reg=1e-10):

    if type(x) is list:
        x = np.vstack(x)

    # Get rotation of origin e to g
    ax, angle = ar.get_axisangle(g)
    Reg = ar.R_from_axis_angle(ax, angle)

    return s2_exp_e(x,reg).dot(Reg.T)

def s2_log(x, g, reg=1e-10):

    if type(x) is list:
        x = np.vstack(x)

    # Get rotation of origin e to g
    ax, angle = ar.get_axisangle(g)
    Reg = ar.R_from_axis_angle(ax, angle)

    return s2_log_e(x.dot(Reg), reg)

def s2_parallel_transport(Xg, g, h, t=1):
    ''' Parallel transport of vectors in X from h to g*t, 0 <= t <= 1
        Implementation is modified version of the one reported in
        Optimization algorithms on Manifolds:
        Xg : array of tangent vectors to parallel tranport  n_data x n_dimT
        g  : base of tangent vectors                        n_dimM
        h  : final point of tangent vectors                 n_dimM
        t  : position on curve between g and h              0 <= t <= 1
    '''
    
    
    # Compute final point based on t
    if t<1:
        ht = s2_exp( s2_log(h, g)*t, g) # current position of h*t in on manifold
    else:
        ht=h
    
    # ----- Compute rotations between different locations:
    
    # Rotation between origin and g
    (ax, angle) = ar.get_axisangle(g)
    Reg = ar.R_from_axis_angle(ax,angle)
    
    # Rotation between origin and ht (in the original atlas)
    (ax, angle) = ar.get_axisangle(ht)
    Reh  = ar.R_from_axis_angle(ax,angle)  # Rotation between final point and origin
    
    # ------ Compute orientation at ht using parallel transpot
    v  = Reg.dot( np.hstack([s2_log(h,g),[0]]) ) # Direction vector in R3
    m  = np.linalg.norm(v)  # Angle of rotation
    
    # Compute Tangential rotation (this is done in n_dimM)
    if m < 1e-10:
        Rgh = np.eye(3)
    else:
        u = (v/m)[:,None]
        g = g[:,None]

        Rgh = (- g*np.sin( m*t )*u.T 
             + u*np.cos( m*t )*u.T
             + ( np.eye(3) - u.dot(u.T) )
            )
        
    # ----- Finally compute rotation compensation to achieve parallel transport:
    Ie = np.eye(3)[:,0:2].T
    Ig   = Ie.dot(Reg.T)                   # Base orientation at g
    Ih   = Ig.dot(Rgh.T)                   # Base orientation at h by parallel transport
    Ie_p = Ih.dot(Reh)                     # Tangent space orientation at origin with parallel transport
    
    # Compute relative rotation:
    R = Ie.dot(Ie_p.T)
    
                   
    # Transform tangent data and return:
    return Xg.dot(R.T)    
    



# General Linear Group:
def GL_exp_e(Hvec,reg=1e-8):
    ''' Exponential map for General Linear Group defined at the Identity'''
    if Hvec.ndim==2:
        res=[]
        for i in range(Hvec.shape[0]):
            res.append( GL_exp_e(Hvec[i,], reg))
        return res
    else:
        H = vec2matrix(Hvec)
        return sp.linalg.expm(H)
    
def GL_exp(Hvec, base, reg=1e-8):
    ''' Exponential map for General Linear group'''
    if Hvec.ndim== 2:
        res=[]
        for i in range(Hvec.shape[0]):
            res.append( GL_exp(Hvec[i,], base, reg))
        return res
    else:
        return base.dot(GL_exp_e(Hvec))
    
def GL_log_e(H, reg=1e-8):
    '''Logarithmic map for General Linear Group'''
    if type(H) is list:
        res=[]
        for item in H:
            res.append( GL_log_e(H, reg))
        return res
    else:
        return  matrix2vec(sp.linalg.logm( H ))

def GL_log(H, base,reg=1e-8):
    ''' Exponential map for PSD matrices'''
    if type(H) is list:
        res = []
        for item in H:
            res.append( GL_log(item, base, reg) ) 
        return np.vstack(res)
    else:
        Xinv = np.linalg.inv(base)
        return matrix2vec(sp.linalg.logm(Xinv.dot( H ) ))

    
def GL_nptoman(data,dim):
    n_elem = data.shape[0]//dim
    
    elemlist = []
    for i in range(n_elem):
        ind = np.arange(dim) + i*dim
        elemlist.append(data[ind,:])
        
    if len(elemlist) ==1:
        return elemlist[0]
    else:
        return elemlist
    
def GL_mantonp(data):
    return np.vstack(data)




# Affine Group:
def getAff(M,v):
    '''Get Affine transformation matrix from M in GL(n) and v in R^n'''
    ndim = len(M) + 1
    H = np.eye(ndim)
    H[:ndim-1, :ndim-1] = M
    H[:ndim-1, ndim-1] = v
    return H

def aff_exp_e(Hvec,reg=1e-4):
    ''' Exponential map for Affine Transformation matrices'''
    if Hvec.ndim== 2:
        res=[]
        for i in range(Hvec.shape[0]):
            res.append( Aff_exp_e(Hvec[i,], reg))
        return res
    else:
        # Get Vector size:
        n = np.sqrt(0.25+len(Hvec)) - 0.5
        if n % 1.0 != 0:
            raise RuntimeError('Invalid vector size')
        else:
            n = int(n)
        
        M = np.zeros( ((n+1),(n+1) ) )
        M[:n,n]  = Hvec[:n]
        M[:n,:n] = vec2matrix(Hvec[n:])
        return sp.linalg.expm(M) 
    
def aff_exp(Hvec, base,reg=1e-8):
    ''' Exponential map for Affine Transformation matrices'''
    if Hvec.ndim ==2:
        res = []
        for i in range(Hvec.shape[0]):
            res.append( aff_exp(Hvec[i,], base, reg) ) 
        return res
    else:
        return base.dot( aff_exp_e(Hvec)  ) 

def aff_log_e(H, reg=1e-8):
    ''' Logarithmic map for Affine Transformation matrices'''
    if type(H) is list:
        res=[]
        for item in H:
            res.append( aff_log_e(H, reg))
        return res
    else:
        n = len(H)-1
        M = sp.linalg.logm(H)
        
        return  np.hstack( (M[:n,n], matrix2vec(M[:n,:n])   ) )   

def aff_log(H, base,reg=1e-8):
    ''' Logarithmic map for Affine Transformation matrices'''
    if type(H) is list:
        res = []
        for item in H:
            res.append( aff_log(item, base, reg) ) 
        return np.vstack(res)
    else:
        return aff_log_e( np.linalg.inv(base).dot(H) ) 


def aff_nptoman(data, n):
    if data.ndim==2:
        res = []
        for i in range(data.shape[0]):
            res.append(aff_nptoman(data[i,:], n) )
        return res
    else:
        R = data[n:].reshape( (n,n) )
        v = data[:n]
        H = np.eye(n+1)
        H[:n,:n] = R
        H[:n,n ] = v
        return H
    
def aff_mantonp(data,n):
    if type(data) is list:
        res = []
        for item in data:
            res.append(SE3_mantonp(item))
        return np.vstack(res)
    else:
        return np.hstack( [data[:n,n], data[:n,:n].reshape(-1 )] )[None,]

# Special Orthogonal Groups SO(2) and SO(3)
def skew2(w):
    """Compute skew symmetric matrix of w"""
    wtilde = np.array([[0, -w],[w,0]])
    return wtilde

def SO2_exp_e(w, reg=1e-4):
    """Identity Exponential map SO(2)"""
    if w.shape[0]==2: 
        res = []
        for i in range(w.shape[0]):
            res.append(SO2_exp_e(w[i,],reg))
        return res
    else:
        w= w[-1]
        return np.array([
                [np.cos(w),-np.sin(w)], 
                [np.sin(w), np.cos(w)]
                ])
    
def SO2_exp(T, base=np.eye(2),reg=1e-8):
    if T.ndim==2: # multiple data points
        # Batch mode:
        res = []
        for i in range(T.shape[0]):
            res.append( SO2_exp(T[i,], base, reg ) )
        return res
    else:
        tmp = SO2_exp_e(T)
        res = base.dot( tmp )
        return res
    
    
def SO2_log_e(R, reg=1e-4) :
    """Identity Logarithmic map SO(2)"""
    if type(R) is list:
        res = []
        for item in R:
            res.append(SO2_log_e(item, reg))
        return res
    else:
        #angle = -sp.linalg.logm(R)[0,1]  # This implementation might have a more efficient computation
        angle = np.arctan2(R[1,0],R[0,0])
        return np.array([angle])

def SO2_log(R,base=np.eye(2),reg=1e-8):
    if type(R) is list:
        res = []
        for e in R:
            res.append( SO2_log(e, base, reg) )
        res = np.vstack(res)
    else:
        Binv = base.T 
        res = SO2_log_e( Binv.dot(R) )
    return res 

def SO2_nptoman(data):
    n_elem = data.shape[0]
    elemlist = []
    for i in range(n_elem):
        R = data[i,].reshape( (2,2) )
        elemlist.append(R)
        
    if len(elemlist) ==1:
        return elemlist[0]
    else:
        return elemlist
    
def SO2_mantonp(data):
    if type(data) is list:
        res = []
        for item in data:
            res.append(SO2_mantonp(item))
        return np.vstack(res)
    else:
        return np.hstack( data[:2,:2].reshape(-1 ) )[None,]


def skew(w):
    """Compute skew symmetric matrix of w"""
    wtilde = np.zeros( (3,3))
    wtilde[0,1] =-w[2]
    wtilde[0,2] = w[1]
    
    wtilde[1,0] = w[2]
    wtilde[1,2] =-w[0]
    
    wtilde[2,0] =-w[1]
    wtilde[2,1] =w[0]
    
    return wtilde

def SO3_exp_e(w, reg=1e-4):
    """Identity Exponential map SO(3)"""
    theta = np.linalg.norm(w)
    if theta < reg:
        # First order taylor expansion:
        return np.eye(3) + skew(w) 
    else:
        wnorm = w/theta
        wtilde = skew(wnorm)
        return np.eye(3) + wtilde*(np.sin(theta)) + (wtilde.dot(wtilde))*(1-np.cos(theta))

def SO3_log_e(R, reg =1e-4):
    theta = np.arccos(np.clip((np.trace(R)-1)/2,-1.0,1.0))
    
    if theta > reg:
        diff = np.array([R[2,1] - R[1,2],
                         R[0,2] - R[2,0],
                         R[1,0] - R[0,1]])
        w = 1/(2*np.sin(theta))*diff
        return theta*w
    else:
        return np.zeros(3)
    
def SO3_exp(T,base=np.eye(3),reg=1e-8):
    if T.ndim==2:
        # Batch mode:
        res = []
        for i in range(T.shape[0]):
            res.append( base.dot( SO3_exp_e(T[i,:] ) ) )
    else:
        res = base.dot( SO3_exp_e(T) )
    return res

def SO3_log(R,base=np.eye(3),reg=1e-8):
    Binv = base.T 
    if type(R) is list:
        res = []
        for e in R:
            res.append( SO3_log_e( Binv.dot(e) ) )
        res = np.vstack(res)
    else:
        res = SO3_log_e( Binv.dot(R) )
        
    return res 

def SO3_nptoman(data):
    n_elem = data.shape[0]
    elemlist = []
    for i in range(n_elem):
        R = data[i,].reshape( (3,3) )
        elemlist.append(R)
        
    if len(elemlist) ==1:
        return elemlist[0]
    else:
        return elemlist
    
def SO3_mantonp(data):
    if type(data) is list:
        res = []
        for item in data:
            res.append(SO3_mantonp(item))
        return np.vstack(res)
    else:
        return np.hstack( data[:3,:3].reshape(-1 ) )



# Special Euclidean Groups SE(2) and SE(3):
# 2-D implementation found at http://www.ethaneade.org/lie.pdf
def Hom(Rij, Oij):
    """Create Homogeneous rotation Matrix from Rotation and Translateion"""
    
    Hij = np.eye(4)
    Hij[0:3,0:3] = Rij
    Hij[0:3, 3]  = Oij
    
    return Hij

def invH(H):
    """Compute invers Homogeneous matrix"""
    n = len(H)-1
    
    A = H[0:n, 0:n].T
    b = -A.dot(H[0:n,n])
    d = 1
    invH = np.eye(n+1)
    invH[0:n,0:n] = A
    invH[0:n, n]  = b
    return invH

def SE2_exp_e(T, reg=1e-4):
    """Identity Exponential map SE(2)"""
    if T.ndim==2:
        res = []
        for i in range(T.shape[0]):
            res.append( SE2_exp_e(T[i,], reg) )
        return res 
    else:
        # Get values:
        w = T[0]
        u = T[1:3]

        # Compute rotation:
        R = SO2_exp_e(T[None,0], reg)

        # Compute Translation:
        if np.abs(w)>reg:
            V = (1/w)*np.array([[np.sin(w)  , -(1-np.cos(w))],
                              [1-np.cos(w), np.sin(w)    ]
                             ])
        else:
            V = np.eye(2)

        t = V.dot(u)
        
        # Construct homogeneous matrix:
        H = np.eye(3)
        H[0:2,0:2] = R
        H[0:2,2]   = t
        return H

def SE2_log_e(H,reg=1e-8):
    if type(H) is list:
        res = []
        for item in H:
            res.append(SE2_log_e(H,reg) )
        return np.vstack(res)
    else:
        # Extract rotation and translations:
        R = H[0:2,0:2]
        t = H[0:2,2]

        # First compute rotation:
        w = SO2_log_e(R)[-1]

        # Extract translation:
        if np.abs(w)>reg:
            A = np.sin(w)/w
            B = (1-np.cos(w))/w
            Vinv = 1/(A**2 + B**2)*np.array([[A, B],[-B,A]])
        else:
            Vinv = np.eye(2)
        u = Vinv.dot(t)
        
        # Return Lie-algebra element:
        return np.hstack([w, u])
    
    
def SE2_exp(T,base=np.eye(3),reg=1e-8):
    if T.ndim==2:
        res = []
        for i in range(T.shape[0]):
            res.append( SE2_exp(T[i,], base, reg) )
        return res 
    else:
        return base.dot(SE2_exp_e(T, reg))

def SE2_log(H,base=np.eye(3),reg=1e-8):
    if type(H) is list:
        res = []
        for item in H:
            res.append(SE2_log(item, base, reg) )
        return np.vstack(res)
    else: 
        Binv= invH(base)
        return SE2_log_e(Binv.dot(H))
   
def SE2_nptoman(data):
    n_elem = data.shape[0]
    elemlist = []
    for i in range(n_elem):
        R = data[i,2:].reshape( (2,2) )
        v = data[i,:2]
        H = np.eye(3)
        H[:2,:2] = R
        H[:2,2 ] = v
        elemlist.append(H)
        
    if len(elemlist) ==1:
        return elemlist[0]
    else:
        return elemlist
    
def SE2_mantonp(data):
    if type(data) is list:
        res = []
        for item in data:
            res.append(SE2_mantonp(item))
        return np.vstack(res)
    else:
        return np.hstack( [data[:2,2], data[:2,:2].reshape(-1 )] )[None,]


def SE3_exp_e(T, reg=1e-4):
    """Identity Exponential map SE(3)"""
    if T.ndim==2:
        res = []
        for i in range(T.shape[0]):
            res.append( SE3_exp_e(T[i,], reg) )
        return res 
    else:
        w = T[0:3]
        v = T[3:6]

        H = np.eye(4)
        if np.linalg.norm(w)<reg:
            # zero rotation, pure translation:
            H[0:3,3] = v
        else:
            # Translation and rotation
            R = SO3_exp_e(w[0:3], reg)
            wtilde = skew(w)
            b = 1/(np.linalg.norm(w)**2)*( (np.eye(3) - R ).dot( wtilde.dot(v) ) 
                                         + w[None:,].dot( v[:,None] )*w)
            H[0:3,0:3] = R
            H[0:3,3]   = b 
        return H

def SE3_log_e(H,reg=1e-8):
    if type(H) is list:
        res = []
        for item in H:
            res.append(SE3_log_e(H,reg) )
        return np.vstack(res)
    else:
        # Extract rotation and translations:
        R = H[0:3,0:3]
        v = H[0:3,3]

        # Perform logarithmic map on SO3 element:
        w = SO3_log_e(R)
        wtilde = skew(w) 
        theta  = np.linalg.norm(w)

        if theta >1e-3:
            div = (2*np.sin(theta) - theta*(1+np.cos(theta)))/(2*(theta**2)*np.sin(theta))
            Ainv = np.eye(3) - 0.5*wtilde + div*wtilde.dot(wtilde)
        else:
            Ainv = np.eye(3)
        return np.hstack([w, Ainv.dot(v)])
    
    
def SE3_exp(T,base=np.eye(4),reg=1e-8):
    if T.ndim==2:
        res = []
        for i in range(T.shape[0]):
            res.append( SE3_exp(T[i,], base, reg) )
        return res 
    else:
        return base.dot(SE3_exp_e(T, reg))

def SE3_log(H,base=np.eye(4),reg=1e-8):
    Binv= invH(base)
    if type(H) is list:
        res = []
        for item in H:
            res.append(SE3_log(item, base, reg) )
        return np.vstack(res)
    else: 
        return SE3_log_e(Binv.dot(H))
   
def SE3_nptoman(data):
    n_elem = data.shape[0]
    elemlist = []
    for i in range(n_elem):
        R = data[i,3:].reshape( (3,3) )
        v = data[i,:3]
        H = np.eye(4)
        H[:3,:3] = R
        H[:3,3 ] = v
        elemlist.append(H)
        
    if len(elemlist) ==1:
        return elemlist[0]
    else:
        return elemlist
    
def SE3_mantonp(data):
    if type(data) is list:
        res = []
        for item in data:
            res.append(SE3_mantonp(item))
        return np.vstack(res)
    else:
        return np.hstack( [data[:3,3], data[:3,:3].reshape(-1 )] )[None,]


# Group of Positive Definite Matrices:
def symmetricmatrix2vec(S):
    ''' Conversion of Symmetric Matrix to equivalent Vector representation'''
    if S.ndim==3:
        res = []
        for i in range(S.shape[0]):
            res.append(symmetricmatrix2vec(S[i,:]))
        return np.vstack(res)
    else:
        (d,_) = S.shape
        v = np.zeros(d+d*(d-1)//2)
        # Diagonal elements:
        v[:d] = np.diag(S)
        
        # Off-diagonal elements:
        row = d
        for i in range(d-1):
            vind = row + d - (1+i)
            v[row:vind] = S[i,(i+1):]*np.sqrt(2)
            row = row + d- (i+1)
        
        return v
    
def vec2symmetricmatrix(v):
    ''' Conversion a vector to a symmetric matrix.'''
    if v.ndim==2:
        res = []
        res = []
        for i in range(S.shape[0]):
            res.append(symmetricmatrix2vec(S[i,:]))
        return res
    else:
        # Find dimension of symmetric matrix:
        t = v.shape[-1];
        n = np.sqrt(0.25 + 2*t) - 0.5
        if (n % 1.0) !=0:
            raise RuntimeError('Dimension of vector is not suitable for construction of symmetric matrix.')
        else:
            n = int(n)

        # Initialize matrix:
        S = np.zeros( (n,n) )

        # Off-diagonal elements
        vind = n;
        for row in range(n-1):
            cols   = np.arange(row+1, n) 
            vrange = np.arange(vind, vind + len(cols)) 
            S[row,cols] = v[vrange]/np.sqrt(2.0)
            
            vind = vrange[-1] + 1;

        S+= S.T               # Other off diagonal
        S+= np.diag(v[0:n]) # Diagonal elements
        
        return S
    
def SPD_exp_e(Hvec,reg=1e-8):
    ''' Exponential map for PSD matrices'''
    H = vec2symmetricmatrix(Hvec)
    
    return sp.linalg.expm(H)
    
def SPD_exp(Hvec,X,reg=1e-8):
    ''' Exponential map for PSD matrices'''
    if Hvec.ndim== 2:
        res=[]
        for i in range(Hvec.shape[0]):
            res.append( SPD_exp(Hvec[i,], X, reg))
        return res
    else:
        # Convert H to matrix:
        H = vec2symmetricmatrix(Hvec)

        # Compute matrix exponential
        (D,V)  = np.linalg.eig(X)
        Xisq = V.dot(np.diag(1/np.sqrt(D))).dot(V.T)
        Xsq = V.dot(np.diag(np.sqrt(D))).dot(V.T)

        arg = Xisq.dot(H.dot(Xisq))
        expX = sp.linalg.expm(arg)
        return Xsq.dot(expX.dot(Xsq))

def SPD_log_e(H, reg=1e-8):
    return SPD_log(H,np.eye(H.shape[0]),reg)

def SPD_log(H,X,reg=1e-8):
    ''' Exponential map for PSD matrices'''
    if type(H) is list:
        res = []
        for item in H:
            res.append( SPD_log(item, X, reg) ) 
        return np.vstack(res)
    else:
        # Convert H to matrix:
        
        # Compute matrix exponential
        (D,V)  = np.linalg.eig(X)
        Xisq = V.dot(np.diag(1/np.sqrt(D))).dot(V.T)
        Xsq = V.dot(np.diag(np.sqrt(D))).dot(V.T)
    
        arg = Xisq.dot(H.dot(Xisq))
        expX = sp.linalg.logm(arg)
        
        return symmetricmatrix2vec(Xsq.dot(expX.dot(Xsq)))

    
def SPD_nptoman(data,dim):
    n_elem = data.shape[0]//dim
    
    elemlist = []
    for i in range(n_elem):
        ind = np.arange(dim) + i*dim
        elemlist.append(data[ind,:])
        
    if len(elemlist) ==1:
        return elemlist[0]
    else:
        return elemlist
    
def SPD_mantonp(data):
    return np.vstack(data)

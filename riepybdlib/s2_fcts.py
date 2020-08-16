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
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import riepybdlib.angular_representations as ar
import riepybdlib.manifold as rm



m_s2 = rm.get_s2_manifold()

def sphere(r,theta,phi):
    return np.array([r*np.sin(theta)*np.cos(phi), 
                     r*np.sin(theta)*np.sin(phi),
                     r*np.cos(theta)])
# 2D Rotation matrix:
fR = lambda theta: np.array([[np.cos(theta),-np.sin(theta)],
                             [np.sin(theta),np.cos(theta)]])



def get_axisangle(d):
    norm = np.sqrt(d[0]**2 + d[1]**2)
    if norm < 1e-6:
        return (np.array([0,0,1]),0)
    else:
        vec = np.array( [-d[1], d[0], 0 ])
        return ( vec/norm,np.arccos(d[2]) )

def get_tangentpoint(Xtg, g, e=m_s2.id_elem):
    m_s2 = rm.get_s2_manifold()
    Xe = np.hstack([Xtg,
                     np.ones((Xtg.shape[0],1))])   # Tangent space at origini
    Xg      = m_s2.action(Xe, e, g)                # Perform action
    return Xg

def geodesic_curve(g,h, n_data = 20, t_st=0, t_end=1):
    ts = (np.linspace(t_st,t_end,n_data, endpoint=True)*m_s2.log(h, g)[:,None]).T
    return m_s2.exp(ts, g)

# Plot functions:
def plot_tangentbase(ax, g, e=np.array([0,0,1]), colors = np.eye(3),vlength=0.4,
                       alpha=1, linestyle='-', label='', angle=0, **kwargs):
    # Get tangent base in G
    R = fR(angle)
    Xg = get_tangentpoint(R.dot(np.eye(2))*vlength,g)
    
    # Plot tangent vector:
    name = ''
    for j in range(2):
        
        if j==1:
            name = label
            
        v = np.vstack([ Xg[j,:], g])
        ax.plot(v[:,0], v[:,1], v[:,2],
                 '-', color=colors[j,], linewidth=2,
                alpha=alpha,
               linestyle=linestyle, label=name,**kwargs)

def plot_tangentdata(Xtg, g, color='red',ax=None, linestyle='-', label='', **kwargs):
    if ax is None:
        ax = plt.gca()
        
    Xtg = get_tangentpoint(Xtg, g)
    # Plot tangent vector:
    v = np.vstack([Xtg, g])
    ax.plot(v[:,0], v[:,1], v[:,2],
             linestyle=linestyle, color=color, 
            linewidth=2, label=label, **kwargs) 

def plot_geodesic(ax, g,h, n_data = 20, t_st=0, t_end=1, linewidth=1,
        linestyle='--',color='grey', **kwargs):
    points = geodesic_curve(g,h,n_data,t_st, t_end)

    ax.plot(points[:,0], points[:,1], points[:,2], linewidth=linewidth,
        linestyle=linestyle, color=color, **kwargs)


def plot_manifold(ax,base=[0,0,1],color=[0.8,0.8,0.8],alpha=0.8,r=0.99, linewidth=0, lim=1.1, n_elems=100,**kwargs):

    u = np.linspace(0, 2 * np.pi, n_elems)
    v = np.linspace(0, np.pi, n_elems)

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color, linewidth=linewidth, alpha=alpha, **kwargs)
    ax.plot(xs=[base[0]], ys=[base[1]], zs=[base[2]],marker='*',color=color)

    
    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])
    ax.set_zlim([-lim,lim])
    

def plot_tangentplane(ax, base, l_vert=1,color='gray',alpha=0.1, linewidth=0, **kwargs):
    # Tangent axis at 0 rotation:
    T0 = np.array([[1,0],
                  [0,1],
                  [0,0]])
    
    # Rotation matrix with respect to zero:
    (axis,ang) = get_axisangle(base)
    R = ar.R_from_axis_angle(axis, -ang)
    
    # Tangent axis in new plane:
    T = R.T.dot(T0)
    
    # Compute vertices of tangent plane at g
    hl = 0.5*l_vert
    X = [[hl,hl],  # p0
         [hl,-hl], # p1
         [-hl,hl], # p2
         [-hl,-hl]]# p3
    X = np.array(X).T
    points = (T.dot(X).T + base).T
    psurf = points.reshape( (-1,2,2))
    
    ax.plot_surface(psurf[0,:],psurf[1,],psurf[2,:],
                    color=color,alpha=alpha,linewidth=0, **kwargs)

def plot_gaussian(ax,mu,sigma, color='red',linewidth=2, linealpha=1,planealpha=0.2,
                        label='', showtangent=True, **kwargs):
    
    # Plot Gaussian
    # - Generate Points @ Identity:
    nbDrawingSeg = 35;    
    t     = np.linspace(-np.pi, np.pi, nbDrawingSeg); 
    R = np.eye(3)
    R[0:2,0:2] = np.real(sp.linalg.sqrtm(1.0*sigma)) # Rotation for covariance
    (axis,angle) = get_axisangle(mu)
    R = ar.R_from_axis_angle(axis,angle).dot(R)      # Rotation for manifold location
    
    points = np.vstack( (np.cos(t), np.sin(t),np.ones(nbDrawingSeg)) )
    points = R.dot(points) 
    
    
    l,= ax.plot(xs=mu[0,None], ys=mu[1,None], zs=mu[2,None], marker='.', 
            color=color,alpha=linealpha, label=label,**kwargs) # Mean

    ax.plot(xs =points[0,:], ys=points[1,:], zs=points[2,:], 
            color=color, 
            linewidth=linewidth, 
            markersize=2, alpha=linealpha,**kwargs) # Contour
    
    if showtangent:
        plot_tangentplane(ax,mu,l_vert=1,color=color,alpha=planealpha, **kwargs)


def loglik_grid(flik, nx,ny):
    # Create mesh grid:
    u = np.linspace(0, 2*np.pi, nx) #+ np.pi/4
    v = np.linspace(0, np.pi, ny )

    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    pgrids = [x,y,z]
    plist= np.hstack( [x.reshape((-1,1)),
                       y.reshape((-1,1)),
                       z.reshape((-1,1))
                        ]
                     )
    # Compute log likelihood for grid:
    lik = flik(plist)
    #lik = -lik/lik.sum()*lik.shape[0]
    likgrid = lik.reshape( (nx,ny) )
    
    return (likgrid, pgrids, plist)
def plot_likelihood2d(flik, base=None, nx=50,ny=50, 
                      ax=None, vmax=0, vmin=-60, showcolorbar=True):
    m_s2 = rm.get_s2_manifold()
    if base==None:
        base= m_s2.id_elem

    # Compute grid:
    (likgrid, pgrids, plist) = loglik_grid(flik, nx,ny)
    
    # Project points on manifold:
    plist_e = m_s2.log(plist, base)
    pgrid_e = plist_e.reshape( (nx,ny,2))

    # Plot points:
    if ax is None:
        fig = plt.figure(figsize=(6,5))
        ax= plt.subplot(111)
    handle = ax.pcolormesh(pgrid_e[:,:,0], pgrid_e[:,:,1], likgrid, 
                       cmap='jet', vmax=vmax, vmin=vmin)
    if showcolorbar:
        cb = plt.colorbar(handle, ticks=[likgrid.min()+1e-1, likgrid.max()])
        cb.set_ticklabels([r'$min$', r'$max$'])

    # Plot minimum:
    lik = likgrid.reshape( (-1))
    i = np.argmax(lik)
    sl = np.ix_(np.abs(lik - lik[i]) < 1e-7, range(plist_e.shape[1]))
    maxs = plist_e[sl]
    
    for i in range(maxs.shape[0]):
        ax.plot(maxs[i,0], maxs[i,1],marker='.',markersize=10, color='black')
        ax.text(maxs[i,0]+0.05, maxs[i,1]+0.05, r'$max$'.format(i),
                )
    
    lim = np.pi
    plt.xlim([-lim, lim]);
    plt.ylim([-lim, lim]);
    plt.axis('equal')
    
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    
    return ax

def plot_likelihood3d(flik, nx=50,ny=50, base=None, ax=None):
    if base is None:
        base = m_s2.id_elem
    
    # Compute grid:
    (likgrid, pgrids, plist) = loglik_grid(flik, nx,ny)
    likgrid = np.prod(likgrid.shape)*likgrid/likgrid.sum() # normalize
    
    # Plot points:
    if ax is None:
        fig = plt.figure(figsize=(5,5))
        ax= plt.subplot(111,projection='3d')
    
    # Create colors:
    cols = np.empty(likgrid.shape, dtype=np.ndarray )
    for i in range(likgrid.shape[0]):
        for j in range(likgrid.shape[1]):
            #tmp = 1 + likgrid[i,j] #np.exp(-likgrid[i,j])
            tmp = 1- likgrid[i,j] #np.exp(-10*likgrid[i,j])
            cols[i,j] = cm.jet(tmp)[0:3]
       
    # Plot surface:
    ax.plot_surface(pgrids[0], pgrids[1], pgrids[2], rstride=4, cstride=4,linewidth=0,
                   facecolors=cols)

    # Plot minimum:
    lik = likgrid.reshape( (-1))
    i = np.argmax(lik)
    sl = np.ix_(np.abs(lik - lik[i]) < 1e-7, range(plist.shape[1]))
    maxs = plist[sl]
    
    lim = 1.1
    ax.set_xlim([-lim, lim]);
    ax.set_zlim([-lim, lim]);
    ax.set_zlim([-lim, lim]);
    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.set_zticks([]);
    
    return ax

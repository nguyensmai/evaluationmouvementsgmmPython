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
import scipy.linalg 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
import matplotlib.cm as cm

# For legend overwrites
from matplotlib.legend_handler import HandlerPatch
from matplotlib.legend import Legend
import matplotlib.patches as mpatches

import riepybdlib.angular_representations as ar
import riepybdlib.s2_fcts as s2_fcts


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


def plotquatCov(ax, q, sigma, pos=np.zeros(3), axlength=1, n_std=1,alpha=1,linewidth=1, covscale=None): 
    '''Plot rotation covariance '''
    cols = np.eye(3)
    Raxis = q.R()

    if covscale is not None:
        print('covscale in plotquatCov(...) has been replaced by n_std. Please specify the number of standard                 deviations using n_std')

    # Plot mean rotation axis, with sufficent length:
    for i in range(3):
        xs = [0,Raxis[0,i]*axlength] + pos[0]
        ys = [0,Raxis[1,i]*axlength] + pos[1]
        zs = [0,Raxis[2,i]*axlength] + pos[2]
        plt.plot(xs=xs, ys=ys, zs=zs, color=cols[i,], alpha=alpha,
                linewidth=3)
    
    
    # Plot Covariance:
    n_drawingsegments = 30
    t = np.linspace(-np.pi, np.pi, n_drawingsegments)
    
    # Compute eigen components of Sigma, and scale according to axis length and number of standard deviations to use:
    UV = sp.linalg.sqrtm(sigma)*axlength*n_std # Eigen components
    
    # Plot each of the axis:
    for i, fR in enumerate([fRx, fRy, fRz]): 
        # Select axis to manipulate:
        ind = [0,1,2]
        ind.remove(i)
        
        # Create contour points on 2-D plane in 3D space:
        tmp = np.zeros((n_drawingsegments, 3))
        tmp[:,ind] =np.vstack([np.cos(t),
                            np.sin(t)]).T
        
        # Define rotation of points:
        # each axis describes the rotation of the other two axis, 
        # we thus need to rotate the covariance by pi/2 of the current axis
        R = fR(np.pi/2)

        # Select Eigen components of remaining axis:
        UVtmp = np.eye(3)
        sl = np.ix_(ind, ind)
        UVtmp[sl] = UV[sl]

        # Rotate contour points to correct position:
        eo = tmp.dot( (R.dot(UVtmp)).T)
    
        # Generate points for covariance
        points = eo.dot(Raxis.T)+ Raxis[:,i]*axlength + pos
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        
        vertices = [[i for i in range(len(x))]]
        tupleList = list(zip(x, y, z))

        poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
        collection = Poly3DCollection(poly3d, linewidths=1, alpha=0.2)
        collection.set_facecolor(cols[i,])

        ax.add_collection3d(collection)
        ax.plot(x,y,z,color=cols[i,],linewidth=linewidth)
        ax.plot(points[:,0], points[:,1], points[:,2],
               color=cols[i,], linewidth=linewidth)


def periodic_clip(val,n_min,n_max):
    ''' keeps val within the range [n_min, n_max) by assuming that val is a periodic value'''
    if val<n_max and val >=n_min:
        val = val
    elif val>=n_max:
        val = val - (n_max-n_min)
    elif val<n_max:
        val = val + (n_max-n_min)
    
    return val
        

def get_points(mu, sigma, n_rings, n_points, n_std=1):
    # Compute eigen components:
    (D0,V0) = np.linalg.eig(sigma)
    U0 = np.real(V0.dot(np.diag(D0)**0.5)*n_std)
     
    # Compute first rotational path
    psi = np.linspace(0,np.pi*2,n_rings, endpoint=True)
    ringpts = np.vstack((np.zeros((1,len(psi))), np.cos(psi), np.sin(psi)))
    
    U = np.zeros((3,3))
    U[:,1:3] = U0[:,1:3]
    ringtmp = U.dot(ringpts)
    
    # Compute touching circular paths
    phi   = np.linspace(0,np.pi,n_points)
    pts = np.vstack((np.cos(phi),np.sin(phi),np.zeros((1,len(phi)))))
    
    xring = np.zeros((n_rings,n_points,3))
    for j in range(n_rings):
        U = np.zeros((3,3))
        U[:,0] = U0[:,0]
        U[:,1] = ringtmp[:,j]
        xring[j,:] = (U.dot(pts).T + mu)
        
    # Reshape points in 2 dimensional array:
    return xring.reshape((n_rings*n_points,3))
        
def tri_ellipsoid(n_rings, n_points):
    ''' Compute the set of triangles that covers a full ellipsoid of n_rings with n_points per ring'''
    tri = []
    for n in range(n_points-1):
        # Triange down
        #       *    ring i+1
        #     / |
        #    *--*    ring i
        tri_up = np.array([n,periodic_clip(n+1,0,n_points),
                          periodic_clip(n+n_points+1,0,2*n_points)])
        # Triangle up
        #    *--*      ring i+1
        #    | / 
        #    *    ring i
        
        tri_down = np.array([n,periodic_clip(n+n_points+1,0,2*n_points),
                          periodic_clip(n+n_points,0,2*n_points)])
        
        tri.append(tri_up)
        tri.append(tri_down)
        
    tri = np.array(tri)
    trigrid = tri
    for i in range(1,n_rings-1):
        trigrid = np.vstack((trigrid,tri+n_points*i))
  
    return np.array(trigrid)



def plot_s2(ax,base=[0,0,1],color=[0.8,0.8,0.8],alpha=0.8,r=0.99, linewidth=0, **kwargs):

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color, linewidth=linewidth, alpha=alpha, zorder=4)
    ax.plot(xs=[base[0]], ys=[base[1]], zs=[base[2]],marker='*',color=color)

def computeCorrelationMatrix(sigma):
    var = np.sqrt(np.diag(sigma))
    return  sigma/var[None,:].T.dot(var[None,:])

def plotCorrelationMatrix(sigma,labels=None,ax=None,labelsize=20):
    cormatrix = computeCorrelationMatrix(sigma)
    n_var = sigma.shape[0]
    if ax==None:
        plt.figure(figsize=(4,3))
        ax = plt.gca();

    if labels is None:
        labels = range(1,n_var+1)
    h = ax.pcolor(cormatrix, cmap='RdBu',vmax=1,vmin=-1)
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.arange(0,n_var)+0.5)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0,n_var)+0.5)
    ax.set_yticklabels(labels)
    ax.tick_params(labelsize=labelsize)
    l = plt.colorbar(h,ticks=[-1,0,1]);            
    l.ax.set_yticklabels([r'$-1$',r'$0$',r'$1$'])
    l.ax.tick_params(labelsize=labelsize)


def distribution_patch(ax, x, mu, var, color=[1, 0, 0], num_std=2, alpha=0.5, linewidth=1, label=''):
    '''
    Function plots the mean and corresponding variance onto the specified axis    

    ax : axis object where the distribution patch should be plotted
    X  : nbpoints array of x-axis values
    Mu : nbpoints array of mean values corresponding to the x-axis values
    Var: nbpoints array of variance values corresponding to the x-axis values

    Author: Martijn Zeestrate, 2015
    '''

    # get number of points:
    npoints = len(x)

    # Create vertices:
    xmsh = np.append(x, x[::-1])
    vTmp = np.sqrt(var) * num_std
    ymsh = np.append(vTmp + mu, mu[::-1] - vTmp[::-1])
    msh = np.concatenate((xmsh.reshape((2 * npoints, 1)), ymsh.reshape((2 * npoints, 1))), axis=1)
    msh = np.concatenate((msh, msh[-1, :].reshape((1, 2))), axis=0)

    # Create codes
    codes = [Path.MOVETO]
    codes.extend([Path.LINETO] * (2 * npoints - 1))
    codes.extend([Path.CLOSEPOLY])

    # Create Path
    path = Path(msh, codes)
    patch = patches.PathPatch(path, facecolor=color, lw=0, edgecolor=color, alpha=alpha)

    # Add to axis:
    ax.add_patch(patch)  # Patch
    ax.plot(x, mu, linewidth=linewidth, color=color, label=label)  # Mean


class AbstractGaussianGraphic(object):
    def __init__(self, ax, mu, sigma):
        mes = """The init class is not implemented. It should add the objects required 
               for GaussianGraphic instance to the axis. The references to 
               these objects should be stored internally such that they
               can be changed by the set_[property]() methods."""
        raise NotImplementedError(mes)

    # Shared properties:
    def set_color(self, color):
        raise NotImplementedError()

    def get_color(self):
        raise NotImplementedError()

    def get_label(self):
        raise NotImplementedError()
    
    def set_data(self, mu, sigma):
        raise NotImplementedError()

    def set_alpha(self, alpha):
        raise NotImplementedError()
        
    # Contour properties
    def set_contouralpha(self, alpha):
        raise NotImplementedError()
        
    def set_contourwidth(self, width):
        raise NotImplementedError()

    # Center properties:
    def set_centersize(self, centersize):
        raise NotImplementedError()




class GaussianPatch3d(AbstractGaussianGraphic):
    def __init__(self, ax, mu, sigma, n_rings=20, n_points=30,
                color='red', alpha=0.5, contourwidth=2,
                n_std=1, label=''):

        # Save properties:
        self._ax = ax
        self._n_points = n_points
        self._n_rings = n_rings
        self._alpha = alpha
        self._linewidth= contourwidth
        self._linecolor = color
        self._patchcolor= color
        self._sigma = sigma
        self._mu = mu
        self._surf = None
        self._label=label
        self._n_std = n_std
        self._update() 

    def _update(self):
        points = get_points(self._mu, self._sigma, self._n_rings, self._n_points, self._n_std)
        triangles= tri_ellipsoid(n_rings=self._n_rings, n_points=self._n_points)
        
        # Plot surface:
        if self._surf is not None:
            self._surf.remove()
        
        # Create new:
        self._surf = self._ax.plot_trisurf(points[:,0],points[:,1],points[:,2],triangles=triangles,
                                     linewidth=self._linewidth,
                                     alpha=self._alpha,
                                     color=self._patchcolor,
                                     edgecolor= self._linecolor)

    # Shared properties:
    def set_color(self, color):
        self._linecolor = color
        self._patchcolor=color
        self._surf.set(edgecolor=color, facecolor=color)

    def get_color(self):
        return self._surf.get_color()

    def get_label(self):
        return self._label

    def set_data(self, mu, sigma):
        self._mu = mu
        self._sigma = sigma
        self._update()

    def set_alpha(self, alpha):
        self._surf.set(alpha=alpha)
        
    # Contour properties
    def set_contouralpha(self, alpha):
        raise NotImplementedError()
        
    def set_contourwidth(self, width):
        self.set(linewidth=width)

    # Center properties:
    def set_centersize(self, centersize):
        raise NotImplementedError()

    def set(self, **kwargs):
        self._surf.set(**kwargs)


class GaussianPatch2d(AbstractGaussianGraphic):
    def __init__(self, ax, mu, sigma, n_segments=35,
                color='red', facealpha=0.5, centersize=8,
                linewidth=1, contouralpha=1, contourwidth=2,
                n_std=1, label=''):
        # Create line object:
        self._n_segments= n_segments
        self._label=label
        self._n_std = n_std
        
        # Generate points
        points = GaussianPatch2d.get_contourpoints(mu,sigma, n_segments, self._n_std)
        
        # polygon
        self._center, = ax.plot(mu[0],mu[1],'.', markersize=centersize, 
                                color=color, alpha=contouralpha)
        self._contour,= ax.plot(points[:,0], points[:,1], 
                             color=color, alpha=contouralpha, 
                             linewidth=contourwidth)
        self._polygon = plt.Polygon(points, color=color, alpha=facealpha)
        ax.add_patch(self._polygon)
        
    @staticmethod
    def get_contourpoints(mu, sigma, n_segments, n_std):
        """Get contour points for a 2d Gaussian distribution"""
        # Compute 
        t = np.linspace(-np.pi, np.pi, n_segments);    
        R = np.real(sp.linalg.sqrtm(n_std*sigma))

        return (R.dot(np.array([[np.cos(t)], [np.sin(t)]]).reshape([2,len(t)])).T + mu)

    def get_label(self):
        return self._label
        
    # Shared properties:
    def set_color(self, color):
        """Set color of Gaussian contour, center and transparant patch."""
        self._center.set_color(color)
        self._contour.set_color(color)
        self._polygon.set_color(color)

    def get_color(self):
        return self._center.get_color()
    
    def set_data(self, mu, sigma):
        """Update data of Gaussian."""
        # Update point
        self._center.set_data(mu)
        
        # Update contour:
        points = GaussianPatch2d.get_contourpoints(mu,sigma, self._n_segments, self._n_std)
        self._contour.set_data(points[:,0], points[:,1])
        self._polygon.set_xy(points)

    def set_alpha(self, alpha):
        self._polygon.set_alpha(alpha)
        
    # Contour properties
    def set_contouralpha(self, alpha):
        """Set transparency of Gaussian"""
        self._contour.set_alpha(alpha)
        self._center.set_alpha(alpha)
        
    def set_contourwidth(self, width):
        """Set width of the contour"""
        self._contour.set_linewidth(width)
        
    # Center properties:
    def set_centersize(self, centersize):
        """Set the size of the center"""
        self._center.set_linewidth(centersize)


class GaussianPatch1d(AbstractGaussianGraphic):
    def __init__(self, ax, mu, sigma, position=0.0,
                 direction='hor',
                 color='red',
                 mirrorpatch=False,
                 scale=1,
                 n_std = 5,
                 contourwidth=2,
                 contouralpha=1,
                 facealpha=0.5,
                label=''):
    
        self._ax = ax             # axis to plot to
        self._mu = mu             # mean (scalar)
        self._sigma = sigma       # variance (scalar)
        self._position = position # Position of the 'baseline'
        self._color=color         # Color
        self._contourwidth = contourwidth
        self._contouralpha = contouralpha
        self._facealpha = facealpha
        self._nstd = n_std        # Number of standard deviations to plot
        self._scale = scale           # Scale of the likelihood values
        self._npoints = int(10*n_std)  # Number of points to plot
        self._direction = direction # Direction: 'vert' or 'hor'
        self._mirrorpatch = mirrorpatch 
        self._label=''
        
        self._patch = None
        self._contour,= self._ax.plot([],[], linewidth= self._contourwidth,
                alpha=self._contouralpha, color=self._color)

        
        self.update()
        
        
    @staticmethod
    def get_contourpoints(mu, sigma, n_std, n_points):
        ext = np.sqrt(sigma)*n_std
        y = np.linspace(-ext,ext,n_points) + mu
        lik = np.exp(-0.5*(y-mu)**2/sigma)
        lik/= lik.max()
        
        return (y, lik)
        
    def update(self):
        
        # Get points:
        (vals, lik)= GaussianPatch1d.get_contourpoints(
                                        self._mu, self._sigma,
                                        self._nstd, self._npoints)
        # Create mesh:
        vals = np.append(vals, vals[-1])


        if self._mirrorpatch:
            lik*=-1
        lik  = np.append(lik, lik[-1])*self._scale + self._position
        
        
        
        msh = np.vstack([vals,lik]).T
        if self._direction=='vert':
            msh= np.roll(msh,1, axis=1)
        
        # Create codes for path:
        codes = [Path.MOVETO]
        codes.extend([Path.LINETO] * (msh.shape[0]-2) )
        codes.extend([Path.CLOSEPOLY])
        
        path = Path(msh, codes)
        # Remove old patch:
        if self._patch is not None:
            self._patch.remove()
        self._patch = patches.PathPatch(path, alpha =self._facealpha, 
                                        color=self._color)
        self._ax.add_patch(self._patch)
        # Update contour:
        if self._direction=='vert':
            self._contour.set_data(lik[:-1], vals[:-1])
        else:
            self._contour.set_data(vals[:-1], lik[:-1])
        #self._ax.plot(vals, lik)
    
        
    def set_color(self, color):
        self._color= color
        self._contour.set_color(color)
        self._patch.set_color(color)
     
    def get_color(self):
        return self._color
    
    def get_label(self):
        return self._label
    
    def set_data(self, mu, sigma):
        self._mu = mu
        self._sigma = sigma
        self.update()

    def set_alpha(self, alpha):
        self._facealpha =alpha
        self._patch.set_alpha(alpha)
        
        
    # Contour properties
    def set_contouralpha(self, alpha):
        self._contouralpha = alpha
        self._contour.set_alpha(alpha)
        
    def set_contourwidth(self, width):
        self._contour.set_linewidth(width)

    # Center properties:
    def set_centersize(self, centersize):
        raise NotImplementedError()

class GaussianGraphicList(AbstractGaussianGraphic,list):
    def __init__(self, *args):
        """Initalize list of Gaussian Graphics"""
        list.__init__(self, *args)
        
    def __getitem__(self,sl):
        return GaussianGraphicList(list.__getitem__(self,sl))
    
    def append(self, item):
        if issubclass(type(item), AbstractGaussianGraphic):
            list.append(self, item)
        else:
            raise TypeError("Item is not of type {0}".format(AbstractGaussianGraphic))

    # Shared properties:
    def set_color(self, color):
        for g in self:
            g.set_color(color)
        self._color=color

    def get_color(self):
        return self[0].get_color()

    def get_label(self):
        return self[0].get_label()

    def set_data(self, mu, sigma):
        for g in self:
            g.set_data(mu, sigma)

    def set_alpha(self, alpha):
        for g in self:
            g.set_alpha(alpha)
        
    # Contour properties
    def set_contouralpha(self, alpha):
        for g in self:
            g.set_contouralpha(alpha)
        
    def set_contourwidth(self, width):
        for g in self:
            g.set_contourwidth(width)

    # Center properties:
    def set_centersize(self, centersize):
        for g in self:
            g.set_centersize(centersize)

class GaussianPatchS2(AbstractGaussianGraphic):
    

    def __init__(self, ax, mu, sigma, n_segments=35,
                color='red', facealpha=0.5, centersize=3,
                contouralpha=1, contourwidth=2,
                centeralpha=None,
                show_tangentplane=True,
                label=''):

        self._label = label
        if centeralpha is None:
            centeralpha=contouralpha
    
        # Plot Gaussian
        # - Generate Points @ Identity:
        self._n_segments = n_segments

        points = GaussianPatchS2.get_contourpoints(mu, sigma, n_segments)
        
        # Graphic elements:
        self._center, = ax.plot(xs=mu[0,None], ys=mu[1,None], zs=mu[2,None], marker='.', 
                                markersize=centersize, 
                                color=color, alpha=centeralpha)
        self._contour,= ax.plot(xs=points[:,0], ys=points[:,1], zs=points[:,2],
                             color=color, alpha=contouralpha, 
                             linewidth=contourwidth)

        if show_tangentplane:
            self._tangentplane = TangentPlanePatchS2(ax, base=mu, shape=(1,1), alpha=0.1, color=color)
        else:
            self._tangentplane = TangentPlanePatchS2(ax, base=mu, shape=(1,1), alpha=0,color=color)


    @staticmethod
    def get_contourpoints(mu, sigma, n_segments):

        # Define segments:
        t = np.linspace(-np.pi, np.pi, n_segments); 
        points = np.vstack( (np.cos(t), np.sin(t),np.ones(n_segments)) )

        # Define rotation to place it in correct position
        R = np.eye(3)
        R[0:2,0:2] = np.real(sp.linalg.sqrtm(1.0*sigma)) # Rotation for covariance
        (axis,angle) = ar.get_axisangle(mu)
        R = ar.R_from_axis_angle(axis,angle).dot(R)      # Rotation for manifold location
        
        points = R.dot(points) 

        return points.T

    # Shared properties:
    def set_color(self, color):
        self._center.set_color(color)
        self._contour.set_color(color)
        self._tangentplane.set_color(color)

    def get_color(self):
        return self._center.get_color()
    
    def set_data(self, mu, sigma):
        # Update point
        self._center.set_data(mu[0], mu[1])
        self._center.set_3d_properties( mu[2] )
        
        # Update contour:
        points = GaussianPatchS2.get_contourpoints(mu,sigma, self._n_segments)
        self._contour.set_data(points[:,0], points[:,1])
        self._contour.set_3d_properties(points[:,2])

        #self._tangentplane.set_base(mu)

    def set_contouralpha(self, alpha):
        self._contour.set_alpha = alpha

    def get_label(self):
        return self._label

    def set_contourwidth(self, width):
        self._contour.set_linewidth(width)

    def set_alpha(self, alpha):
        self.set_contouralpha(alpha)
        self.set_centeralpha(alpha)

    def set_centeralpha(self, alpha):
        self._center.set_alpha = alpha

    def set_centersize(self, centersize):
        self._center.set_markersize(centersize)


class PegPatch(object):
    
    def __init__(self, ax, A, b, color='red', 
                 scale=.1, alpha=1):
        """Create plot patch of U-shape."""
        self._scale = scale
        self._A = A
        self._b = b
        self._color = color
        self._alpha = alpha
        self._linewidth = 100
        
        points = PegPatch.get_pegpoints(A,b, scale)
        self._line, = ax.plot(points[:,0], points[:,1], 
                              color=self._color, linewidth=self._linewidth*self._scale,
                             alpha=self._alpha)
    
    @staticmethod
    def get_pegpoints(A, b, scale):
        """Get plot points for the peg shape."""
        points = scale*np.array([[-1,-1, 1, 1],
                  [ 3, -0.4, -0.4, 3],                  
                 ]).T
        return points.dot(A.T) + b
    
    def _update(self):
        """Update plot"""
        points = PegPatch.get_pegpoints(self._A,self._b, self._scale)
        self._line.set_data(points[:,0], points[:,1])
        self._line.set_linewidth(self._scale*self._linewidth)
    
    def set_alpha(self, alpha):
        """Set transparency parameter alpha"""
        self._line.set_alpha(alpha)
        
    def set_scale(self, scale):
        """Set scale of U-shape"""
        self._scale = scale
        self._update()
        
    def set_pose(self, A, b):
        """Set pose of of the U-shape"""
        self._A = A
        self._b = b
        self._update()


class PegPatch(object):
    
    def __init__(self, ax, A, b, color='red', 
                 scale=.1, alpha=1):
        """Create plot patch of U-shape."""
        self._scale = scale
        self._A = A
        self._b = b
        self._color = color
        self._alpha = alpha
        self._linewidth = 100
        
        points = PegPatch.get_pegpoints(A,b, scale)
        self._line, = ax.plot(points[:,0], points[:,1], 
                              color=self._color, linewidth=self._linewidth*self._scale,
                             alpha=self._alpha)
    
    @staticmethod
    def get_pegpoints(A, b, scale):
        """Get plot points for the peg shape."""
        points = scale*np.array([[-1,-1, 1, 1],
                  [ 3, -0.4, -0.4, 3],                  
                 ]).T
        return points.dot(A.T) + b
    
    def _update(self):
        """Update plot"""
        points = PegPatch.get_pegpoints(self._A,self._b, self._scale)
        self._line.set_data(points[:,0], points[:,1])
        self._line.set_linewidth(self._scale*self._linewidth)
    
    def set_alpha(self, alpha):
        """Set transparency parameter alpha"""
        self._line.set_alpha(alpha)
        
    def set_scale(self, scale):
        """Set scale of U-shape"""
        self._scale = scale
        self._update()
        
    def set_pose(self, A, b):
        """Set pose of of the U-shape"""
        self._A = A
        self._b = b
        self._update()


class PegPatchS2(object):
    
    def __init__(self, ax, A, b, color='red', 
                 scale=.07, alpha=1):
        """Create plot patch of U-shape."""
        self._scale = scale
        self._A = A
        self._b = b
        self._color = color
        self._alpha = alpha
        self._linewidth = 70
        
        points = PegPatchS2.get_pegpoints(A,b, scale)
        self._line, = ax.plot(points[:,0], points[:,1], points[:,2],
                              color=self._color, linewidth=self._linewidth*self._scale,
                             alpha=self._alpha)
    

    @staticmethod
    def get_pegpoints(A, b, scale):
        """Get plot points for the peg shape."""
        points = scale*np.array([[-1,-1, 1, 1],
                  [ 3, -0.4, -0.4, 3],                  
                 ]).T
        points = points.dot(A.T)
        return s2_fcts.get_tangentpoint(points, b)
    
    def _update(self):
        """Update plot"""
        points = PegPatchS2.get_pegpoints(self._A,self._b, self._scale)
        self._line.set_data(points[:,0], points[:,1])
        self._line.set_3d_properties(points[:,2])
        self._line.set_linewidth(self._scale*self._linewidth)
    
    def set_alpha(self, alpha):
        """Set transparency parameter alpha"""
        self._line.set_alpha(alpha)
        
    def set_scale(self, scale):
        """Set scale of U-shape"""
        self._scale = scale
        self._update()
        
    def set_pose(self, A, b):
        """Set pose of of the U-shape"""
        self._A = A
        self._b = b
        self._update()


class TangentPlanePatchS2(object):
    def __init__(self, ax, base, shape=(1,1), color='gray', alpha=0, edgewidth=0):

        self._ax = ax
        self._base = base
        self._shape = shape
        self._alpha = alpha
        self._edgewidth = edgewidth
        self._color = color

        # Create surface:
        self.redraw()

    def update_surface(self):
        psurf = TangentPlanePatchS2.get_surfacepoints(base, shape)
        self._surf.set_data(psurf[:,0], psurf[:,1])
        self._surf.set_3d_properties( psurf[:,2])
        
    def redraw(self):
        if hasattr(self, '_surf'):
            self._surf.remove()

        psurf = TangentPlanePatchS2.get_surfacepoints(self._base, self._shape)
        self._surf = self._ax.plot_surface(psurf[0,:],psurf[1,],psurf[2,:],
                        color=self._color,alpha=self._alpha,linewidth=self._edgewidth)

    @staticmethod
    def get_surfacepoints(base, shape):
        # Tangent axis at 0 rotation:
        T0 = np.array([[1,0],
                      [0,1],
                      [0,0]])
        
        # Rotation matrix with respect to zero:
        (axis,ang) = ar.get_axisangle(base)
        R = ar.R_from_axis_angle(axis, -ang)
        
        # Tangent axis in new plane:
        T = R.T.dot(T0)
        
        # Compute vertices of tangent plane at g
        dx = 0.5*shape[0]
        dy = 0.5*shape[1]
        X = [[ dx, dy], # p0
             [ dx,-dy], # p1
             [-dx, dy], # p2
             [-dx,-dy]] # p3
        X = np.array(X).T
        points = (T.dot(X).T + base).T
        psurf = points.reshape( (-1,2,2))
        return psurf 

    def set_shape(self, shape):
        raise NotImplementedError()

    def set_base(self, shape):
        raise NotImplementedError()

    def set_color(self, color):
        self._surf.set_color(color)

    def set_alpha(self, alpha):
        self._alpha = alpha
        self.redraw()



class FrameOfReference3D(object):
    
    def __init__(self, ax, pos, R, scale=.07, alpha=1, linewidth=1):
        """Create plot patch of U-shape."""

        self._linewidth=3
        self._alpha = alpha
        self._pos = pos
        self._R = R
        self._scale = scale
        
        cols = np.eye(3)

        # Create lines:
        self._lines = []
        for i in range(3):
            tmpl, = ax.plot([],[],[],
                            color=cols[i,], linewidth=self._linewidth,
                            alpha=self._alpha)
            self._lines.append(tmpl)
        # Draw data:
        self._update()

    
    @staticmethod
    def get_linepoints(pos, R, scale):
        """Get plot points for the peg shape."""
        
        points = [] 
        for i in range(3):
            xs = [0, R[0,i]*scale] + pos[0]
            ys = [0, R[1,i]*scale] + pos[1]
            zs = [0, R[2,i]*scale] + pos[2]
            points.append([xs, ys, zs])

        return points
    
    def _update(self):
        """Update plot"""
        points = FrameOfReference3D.get_linepoints(self._pos, self._R, self._scale)
        # Create lines:
        for i in range(3):
            [xs, ys, zs] = points[i]
            self._lines[i].set_data(xs, ys)
            self._lines[i].set_3d_properties(zs)
    
    def set_alpha(self, alpha):
        """Set transparency parameter alpha"""
        self._line.set_alpha(alpha)
        
    def set_scale(self, scale):
        """Set scale of U-shape"""
        self._scale = scale
        self._update()

    def set_linewidth(self, linewidth):
        """Set linewidth of axes"""

        for i in range(3):
            self._lines[i].set_linewidth(linewidth)
        
    def set_pose(self, pos, R):
        """Set pose of of the U-shape"""
        self._pos = pos
        self._R = R
        self._update()

class FrameOfReference2D(object):
    
    def __init__(self, ax, pos, R, scale=1, alpha=1, linewidth=1):
        """Create plot patch of U-shape."""

        self._linewidth=3
        self._alpha = alpha
        self._pos = pos
        self._R = R
        self._scale = scale
        
        cols = np.eye(3)

        # Create lines:
        self._lines = []
        for i in range(2):
            tmpl, = ax.plot([],[],
                            color=cols[i,], linewidth=self._linewidth,
                            alpha=self._alpha)
            self._lines.append(tmpl)
        # Draw data:
        self._update()

    
    @staticmethod
    def get_linepoints(pos, R, scale):
        """Get plot points for the peg shape."""
        
        points = [] 
        for i in range(2):
            xs = [0, R[0,i]*scale] + pos[0]
            ys = [0, R[1,i]*scale] + pos[1]
            points.append([xs, ys])

        return points
    
    def _update(self):
        """Update plot"""
        points = FrameOfReference2D.get_linepoints(self._pos, self._R, self._scale)
        # Create lines:
        for i in range(2):
            [xs, ys] = points[i]
            self._lines[i].set_data(xs, ys)
    
    def set_alpha(self, alpha):
        """Set transparency parameter alpha"""
        self._line.set_alpha(alpha)
        
    def set_scale(self, scale):
        """Set scale of U-shape"""
        self._scale = scale
        self._update()

    def set_linewidth(self, linewidth):
        """Set linewidth of axes"""

        for i in range(2):
            self._lines[i].set_linewidth(linewidth)
        
    def set_pose(self, pos, R):
        """Set pose of of the U-shape"""
        self._pos = pos
        self._R = R
        self._update()

class S2ManifoldPatch(object):
    
    def __init__(self, ax, color=[0.7, 0.7, 0.7], alpha=0.6, r=1.0,
                linewidth=0, n_segments=100):
        
        self._ax = ax
        self._n_segments=n_segments
        self._color = color
        self._linewidth = linewidth
        self._alpha = alpha
        self._r = r
        (self._x, self._y, self._z) = S2ManifoldPatch.get_surfacepoints(self._r, self._n_segments)
        self.update()
        self.set_axis()
        
    def set_axis(self,lim=[-1,1]):
        self._ax.set_xlim(lim)
        self._ax.set_ylim(lim)
        self._ax.set_zlim(lim)
        self._ax.axis('off')

    @staticmethod
    def get_surfacepoints(r, n_segments):

        u = np.linspace(0, 2 * np.pi, n_segments)
        v = np.linspace(0, np.pi, n_segments)

        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        return x,y,z
    
    def update(self):
        self._surf = self._ax.plot_surface(self._x, self._y, self._z,  
                        rstride=4, cstride=4, 
                        color=self._color, 
                        linewidth=self._linewidth, 
                        alpha= self._alpha, 
                        zorder=4)
        #ax.plot(xs=[base[0]], ys=[base[1]], zs=[base[2]],marker='*',color=color)
        
    def set_alpha(self, alpha):
        self._alpha = alpha
        self._surf.set_alpha(alpha)
        
    def set_linewidth(self, linewidth):
        self._linewidth = linewidth
        self._surf.set_linewidth(linewidth)
    


def plot_gaussian_2d(mu, sigma, ax=None, 
        linewidth=1, alpha=0.5, color=[0.6,0,0], label='', markersize=2, **kwargs):
    ''' This function displays the parameters of a Gaussian .'''
    print('The function plot_gaussian_2d(...), will be removed in future releases.')
    # Create axis if not specified
    if ax is None:
        ax = plt.gca();

    return GaussianPatch2d(ax, mu, sigma, linewidth=linewidth, alpha=alpha, 
                           color=color, centersize=markersize)

def plotRotation(ax, q, pos=np.zeros(3), length=1, alpha=1, color=None, label='', **kwargs):
    print('The function plotRotation(...), will be removed in future releases.')
    return FrameOfReference3D(ax, pos=np.zeros(3), R=q.R(), scale=length, alpha=alpha)


def plot_gaussian_s2(ax,mu,sigma, color='red',linewidth=2, linealpha=1,planealpha=0.2,
                        label='', showtangent=True, **kwargs):
    print('The function plot_gaussian_s2(...), will be removed in future releases.')
    return GaussianPatchS2(ax, mu, sigma, color=color, facealpha=planealpha, centersize=2,
                contouralpha=linealpha, contourwidth=linewidth, show_tangentplane=showtangent)
                    
def plot_tangentplane_s2(ax, base, l_vert=1,color='gray',alpha=0.1, linewidth=0, **kwargs):
    print('The function plot_gaussian_s2(...), will be removed in future releases.')
    return TangentPlanePatchS2(ax, base, shape=(1,1), color=color, alpha=alpha, edgewidth=linewidth)


def plot_gaussian_3d(mu, sigma, ax=None, n_points=30, n_rings=20, 
                     linewidth=0, alpha=0.5, color=[0.6, 0, 0], label='', **kwargs):
    ''' Plot 3d Gaussian'''
    print('The function plot_gaussian_3d(...), will be removed in future releases.')
    # Create axis if not provided
    if ax is None:
        ax = plt.gca();
    return GaussianPatch3d(ax, mu, sigma, n_rings=n_rings, n_points=n_points,
                color=color, alpha=alpha, contourwidth=linewidth,
                n_std=1, label=label)



class GaussianLegendHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        
        center = 0.5 * width - 0.5 * x0, 0.5 * height - 0.5 * y0
        patch = mpatches.Ellipse(xy=center, width=width + x0,
                            height=height + y0,
                            transform=handlebox.get_transform(),
                            color=orig_handle.get_color(),
                            linewidth=1) 
        
        
        handlebox.add_artist(patch)
        return patch
Legend.update_default_handler_map({AbstractGaussianGraphic: GaussianLegendHandler()})

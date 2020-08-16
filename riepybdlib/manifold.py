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
from copy import deepcopy
from copy import copy

import riepybdlib.angular_representations as ar
from riepybdlib.mappings import *


# ------------------------ Classes --------------------------------
class Manifold(object):
    """ Riemannian Manifold Class

    To specify a 'root' manifold one needs to provide:
    n_dimM : Dimension of Manifold space
    n_dimT : Dimension of Tangent space
    exp    : Exponential function that maps elemquifoldents from the tangent space to the manifold
    log    : Logarithmic function that maps elements from the manifold to the tangent space
    init   : (optional) an initialiation element
    And optionally:
    f_nptoman: A function that defines how a np array of numbers is tranformed into a manifold element
    If these parameters are not provides, some of the functions of the manifold don't work

    Alternatively one can create a composition of root manifolds by providing a list of manifolds
    """

    def __init__(self,
                 n_dimM=None, n_dimT=None,
                 exp_e=None, log_e=None, id_elem=None,
                 name='', f_nptoman=None,
                 f_mantonp=None, manlist=None,
                 f_action=None,
                 f_parallel_transport=None,
                 exp=None, log=None
                 ):

        # Check input
        if (((n_dimM is None) and (n_dimT is None) and
             (exp_e is None) and (log_e is None) and (id_elem is None) and
             (f_action is None)
        )
                and
                (manlist is None)
        ):
            raise RuntimeError('Improper Manifold specification, either specify root manifold by providing atleast' +
                               'n_dimM, n_dimT, exp, log, init, fcocM, fcocT or provide a manifold list')

        if manlist is not None:
            # None-root manifold (i.e. a product of manifolds)
            self.__manlist = []
            self.n_dimT = 0
            self.n_dimM = 0
            id_list = []  # List of identity elements
            name = ''
            for i, man in enumerate(manlist):
                # Check existance of manifold and add to list
                if type(man) is Manifold:
                    self.__manlist.append(man)
                else:
                    raise RuntimeError('Non-manifold type found in manifold list.')

                # Gather properties of cartesian product of manifolds:
                self.n_dimT += man.n_dimT
                self.n_dimM += man.n_dimM
                id_list.append(man.id_elem)
                # combine names:
                if i > 0:
                    name = '{0} x {1}'.format(name, man.name)
                else:
                    name = '{0}'.format(man.name)
            self.name = name

            # Assign functions to manifold:
            if len(manlist) > 1:
                # If there are more manifold in the list, assign the public manifold functions:
                self.id_elem = tuple(id_list)
                self.__fnptoman = self.np_to_manifold
                self.__fmantonp = self.manifold_to_np
                self.__fparalleltrans = self.parallel_transport
                self.__faction = self.action

                # Mapping functions are defined recursively through the exp, and log functions of this manifold
                self.__fexp = self.exp
                self.__flog = self.log
            else:
                # If there is only one manifold in the list, we need to use the dedicated functions
                self.id_elem = manlist[-1].id_elem
                self.__fnptoman = manlist[-1].np_to_manifold
                self.__fmantonp = manlist[-1].manifold_to_np
                self.__fparalleltrans = manlist[-1].parallel_transport
                self.__faction = manlist[-1].action

                # Mapping functions are defined recursively through the exp, and log functions of this manifold
                self.__fexp = manlist[-1].exp
                self.__flog = manlist[-1].log



        else:
            # Root manifold:
            if n_dimT is None:
                n_dimT = n_dimM

            if f_nptoman is None:
                # Specify the function as a simple one-to-one mapping:
                f_nptoman = lambda data: data  # if type(data) is np.ndarray else np.array([data])
                self.__fnptoman = f_nptoman
            else:
                self.__fnptoman = f_nptoman

            if f_mantonp is None:
                # Specify the function as a simple one-to-one mapping:
                f_mantonp = lambda data: data  # if type(data) is np.ndarray else np.array([data])
                self.__fmantonp = f_mantonp
            else:
                self.__fmantonp = f_mantonp

            # Assign action functions:
            self.__faction = f_action
            self.__fparalleltrans = f_parallel_transport

            # Create the log and exp mappings using the base functions log and exp
            # Check if the log and exp maps are provided (this could yield faster computation)
            # Otherwise create them using the action function:
            if exp is None:
                self.__fexp = lambda x_tan, base, reg: f_action(exp_e(x_tan, reg), id_elem, base)
            else:
                self.__fexp = exp

            if log is None:
                self.__flog = lambda x, base, reg: log_e(f_action(x, base, id_elem), reg)
            else:
                self.__flog = log

            # The manifold list only consists of the Manifold itself:
            self.__manlist = [self]

            self.id_elem = id_elem
            self.n_dimT = n_dimT
            self.n_dimM = n_dimM
            self.name = name

        self.n_manifolds = len(self.__manlist)

    def __mul__(self, other):
        '''Implementation of the Cartesian products of Manifolds'''
        manlist = []
        for _, item in enumerate(self.__manlist):
            manlist.append(item)
        for _, item in enumerate(other.__manlist):
            manlist.append(item)

        return Manifold(manlist=manlist)

    def exp(self, g_tan, base=None, reg=1e-10):
        ''' Manifold Exponential map

        Arguments
        g_tan: n_data x n_dimT array of tangent vectors
        base : tangent space base 

        Keyword arguments:
        reg  : Regularization term for exponential map
        '''

        # If base is not specified, use the id-element of the manifold:
        if base is None:
            base = self.id_elem

        # Single Manifolds will not have base in tuple form, adopt:
        if type(base) is not tuple:
            base = tuple([base])

        # If g_tan only has a single dimension, we assume it consists of one data point
        d_added = False
        if g_tan.ndim == 1:
            g_tan = g_tan[None, :]
            d_added = True;

        tmp = []
        ind = 0
        for i, man in enumerate(self.__manlist):
            g_tmp = man.__fexp(g_tan[:, np.arange(man.n_dimT) + ind], base[i], reg=reg)
            if d_added:
                # Remove additional dimension
                tmp.append(g_tmp[0])
            else:
                tmp.append(g_tmp)
            ind += man.n_dimT

        if len(self.__manlist) == 1:
            return tmp[0]
        else:
            return tuple(tmp)

    def log(self, g, base=None, reg=1e-10):
        '''Manifold Logarithmic map

        Arguments
        g    : tuple of list 
        base : tangent space base 

        Keyword arguments:
        reg  : Regularization term for exponential map
        
        '''
        # Single Manifolds will not have base in tuple form, adopt:
        if base is None:
            base = self.id_elem
        if type(g) is not tuple:
            g = tuple([g])
        if type(base) is not tuple:
            base = tuple([base])

        g_tan = []
        for i, man in enumerate(self.__manlist):
            g_tan.append(man.__flog(g[i], base[i], reg=reg))

        return np.hstack(g_tan)

    def action(self, X, g, h):
        ''' Create manifold elements Y that have a relation with h and elements X have with g'''
        # Single Manifolds will not have base in tuple form, adopt:
        if type(g) is not tuple:
            g = tuple([g])
        if type(h) is not tuple:
            h = tuple([h])
        if type(X) is not tuple:
            X = tuple([X])

        # Perform actions
        Y = []
        for i, man in enumerate(self.__manlist):
            if man.__faction is None:
                raise RuntimeError('Action function not specified for manifold {0}'.format(man.name))
            else:
                Y.append(man.__faction(X[i], g[i], h[i]))

        if len(self.__manlist) == 1:
            return Y[0]
        else:
            return tuple(Y)

    def parallel_transport(self, Xg, g, h, t=1):
        ''' Create manifold Parallel transport Xg from tangent space at g, to tangent space at h'''
        # Single Manifolds will not have base in tuple form, adopt:
        if type(g) is not tuple:
            g = tuple([g])
        if type(h) is not tuple:
            h = tuple([h])

        d_added = False;
        if Xg.ndim == 1:
            Xg = Xg[None, :]
            d_added = True;

        # Perform actions
        Xh = np.zeros(Xg.shape)
        ind = 0
        for i, man in enumerate(self.__manlist):
            if man.__fparalleltrans is None:
                raise RuntimeError('Parallel transport not specified for manifold {0}'.format(man.name))
            else:
                sl = np.arange(man.n_dimT) + ind
                Xh[:, sl] = man.__fparalleltrans(Xg[:, sl], g[i], h[i], t)

            ind += man.n_dimT

        if d_added:
            return Xh[0, :]
        else:
            return Xh

    def np_to_manifold(self, data):
        '''Transfrom nparray in a tuple of manifold elements
        data: n_data x n_dimM  array
        output: tuple( M1, M2, ..., Mn), in which M1 is a list of manifold elements
       
        '''
        tmp = []
        ind = 0
        for j, man in enumerate(self.__manlist):
            if data.ndim == 1:
                tmp.append(man.__fnptoman(data[np.arange(man.n_dimM) + ind]))
            else:
                tmp.append(man.__fnptoman(data[:, np.arange(man.n_dimM) + ind]))
            ind += man.n_dimM

        if j == 0:
            # Single manifold, remove the list structure:
            return tmp[0]
        else:
            # Combined manifold, transform the list in a tuple
            return tuple(tmp)

    def manifold_to_np(self, data):
        ''' Transform manifold data into a numpy array'''
        # Ensure that we can handle both single samples and arrays of samples:
        np_list = []
        if len(self.__manlist) == 1:
            npdata = self.__fmantonp(data)
        else:
            for j, man in enumerate(self.__manlist):
                tmp = man.__fmantonp(data[j])
                # if (man.n_dimM == 1) and (tmp.ndim == 1):
                #    tmp = tmp[:,None]
                np_list.append(tmp)
            npdata = np.hstack(np_list)

        return npdata

    def swapto_tupleoflist(self, data):
        ''' Swap data from list of tuples to tuple of lists'''
        if (type(data) is tuple or
                type(data) is type(self.id_elem)):
            tupleoflist = data
        elif type(data) is list:
            # Data is list of individual tuples:
            #          1           ---          n_data
            # [ (  submanifold_1 )        (  submanifold_1 ) ]
            # [ (  |             ), ---  ,(  |             ) ]
            # [ (  submanifold_N )        (  submanifold_N ) ]
            # We swap to list:
            npdata = []
            for i, elem in enumerate(data):
                npdata.append(self.manifold_to_np(elem))
            npdata = np.vstack(npdata)
            if npdata.ndim == 2 and npdata.shape[0] == 1:
                npdata = npdata[0, :]  # drop dimension
            tupleoflist = self.np_to_manifold(npdata)
        else:
            raise TypeError('Unknown type {0} encoutered for swap'.format(type(data)))

        return tupleoflist

    def swapto_listoftuple(self, data):
        ''' Swap data from list of tuples to tuple of lists'''
        if type(data) is list:
            listoftuple = data
        elif type(data) is np.ndarray and self.n_manifolds == 1:
            if data.shape == self.id_elem.shape:
                # This is a single element, extend dimension
                data = data[None, :]
            elif data.ndim == 1:
                data = data[:, None]
            # Transform entries in list:
            listoftuple = list(data)
        elif type(data) is tuple:
            # Data is tuple of lists
            # ( [nbdata x submanifold_1]  )
            # ( [ |                    ]  )
            # ( [nbdata x submanifold_N]  )

            dofnlist = []  # List of D-manifolds each containing N elements
            for i, elem in enumerate(list(data)):
                nlist = self.get_submanifold(i).swapto_listoftuple(elem)
                dofnlist.append(nlist)

            listoftuple = []
            for n in range(len(dofnlist[0])):
                elem = tuple([dofnlist[d][n] for d in range(self.n_manifolds)])
                listoftuple.append(elem)
        elif type(data) is type(self.id_elem):
            listoftuple = [data]
        else:
            raise TypeError('Unknown type {0} encoutered for swap'.format(type(data)))

        return listoftuple

    def swap_btwn_tuplelist(self, data):
        ''' Swap between tuple of data points and list of tuples'''
        if type(data) is list:
            return self.swapto_tupleoflist(data)
        elif type(data) is tuple:
            return self.swapto_listoftuple(data)
        elif type(data) is np.ndarray:
            return data
        else:
            raise RuntimeError('Unknown type {0} encoutered for swap'.format(type(data)))

    def get_submanifold(self, i_man):
        ''' Returns a manifold that of the requested indices
        i_man : (list of) manifold index
        
        '''

        if type(i_man) is list:
            # List of indices requested:
            manlist = []
            for _, ind in enumerate(i_man):
                manlist.append(self.get_submanifold(ind))
            # if len(manlist)==1:
            #   return manlist[-1] # Return single manifold
            # lse:
            return Manifold(manlist=manlist)  # Return new combination of manifolds
        else:
            # Check input
            if i_man > len(self.__manlist):
                raise RuntimeError('index {0} exceeds number of submanifolds.'.format(i_man))
            # Return requested sub-manifold
            return self.__manlist[i_man]

    def get_tangent_indices(self, i_man):
        '''Get the tangent space indices for a (list of) manifold(s)
        i_man : (list of) manifold index
        '''
        if type(i_man) is list:
            # List of indices requested:
            indlist = []
            for _, ind in enumerate(i_man):
                # Get manifold indices:
                tmp = self.get_tangent_indices(ind)

                # Copy indices:
                for _, i in enumerate(tmp):
                    indlist.append(i)
            return indlist
        else:
            # Check input
            if i_man > len(self.__manlist):
                raise RuntimeError('index exceeds number of submanifolds')

            # Create & return range
            st_in = 0
            for i in range(0, i_man):
                st_in += self.__manlist[i].n_dimT
            return np.arange(st_in, st_in + self.__manlist[i_man].n_dimT).tolist()

    def n_manifolds(self):
        return len(self.__manlist)


# Define two standard manifolds:
def get_euclidean_manifold(n_dim, name='Euclidean Manifold'):
    return Manifold(n_dimM=n_dim, n_dimT=n_dim,
                    exp_e=eucl_exp_e, log_e=eucl_log_e, id_elem=np.zeros(n_dim),
                    name=name,
                    f_action=eucl_action,
                    f_parallel_transport=eucl_parallel_transport
                    )


def get_quaternion_manifold(name='Quaternion Manifold'):
    return Manifold(n_dimM=4, n_dimT=3,
                    exp_e=quat_exp_e, log_e=quat_log_e, id_elem=quat_id,
                    name=name,
                    f_nptoman=ar.Quaternion.from_nparray,
                    f_mantonp=ar.Quaternion.to_nparray_st,
                    f_action=quat_action,
                    f_parallel_transport=quat_parallel_transport,
                    exp=quat_exp, log=quat_log  # Add optional non-base maps that to provide more efficient computation
                    )


def get_s2_manifold(name='S2', fnptoman=None, fmantonp=None):
    return Manifold(n_dimM=3, n_dimT=2,
                    exp_e=s2_exp_e, log_e=s2_log_e, id_elem=s2_id,
                    name=name,
                    f_nptoman=fnptoman,
                    f_mantonp=fmantonp,
                    f_action=s2_action,
                    f_parallel_transport=s2_parallel_transport,
                    exp=s2_exp, log=s2_log  # Add optional non-base maps that to provide more efficient computation
                    )


def SO2_parallelTransp(xtan, a, b, t):
    return xtan


def get_SO2(name='SO(2)'):
    return Manifold(n_dimM=4, n_dimT=1,
                    exp_e=SO2_exp_e, log_e=SO2_log_e,
                    log=SO2_log, exp=SO2_exp,
                    id_elem=np.eye(2),
                    name=name,
                    f_parallel_transport=SO2_parallelTransp,
                    f_nptoman=SO2_nptoman, f_mantonp=SO2_mantonp,
                    )


def get_SO3(name='SO(3)'):
    return Manifold(n_dimM=9, n_dimT=3,
                    exp_e=SO3_exp_e, log_e=SO3_log_e,
                    log=SO3_log, exp=SO3_exp,
                    id_elem=np.eye(3),
                    name=name,
                    f_nptoman=SO3_nptoman, f_mantonp=SO3_mantonp,
                    )


def get_SE3(name='SE(3)'):
    return Manifold(n_dimM=12, n_dimT=6,
                    exp_e=SE3_exp_e, log_e=SE3_log_e,
                    log=SE3_log, exp=SE3_exp,
                    id_elem=np.eye(4),
                    name=name,
                    f_nptoman=SE3_nptoman, f_mantonp=SE3_mantonp,
                    )


def get_SE2(name='SE(2)'):
    return Manifold(n_dimM=6, n_dimT=3,
                    exp_e=SE2_exp_e, log_e=SE2_log_e,
                    log=SE2_log, exp=SE2_exp,
                    id_elem=np.eye(3),
                    name=name,
                    f_nptoman=SE2_nptoman, f_mantonp=SE2_mantonp,
                    )


def get_SPDn(n):
    return Manifold(n_dimM=(n + 1) * n // 2, n_dimT=(n + 1) * n // 2,
                    exp_e=SPD_exp_e, exp=SPD_exp,
                    log_e=SPD_log_e, log=SPD_log,
                    id_elem=np.eye(n), name='SPD({0}'.format(n),
                    f_nptoman=lambda data: SPD_nptoman(data, n), f_mantonp=SPD_mantonp,
                    )


# def get_GLn(n):
#      return Manifold(n_dimM=n**2, n_dimT=n**2,
#                         exp_e= GL_exp_e, log_e = GL_log_e,
#                         exp  = GL_exp  , log   = GL_log,
#                         id_elem=np.eye(n), name='GL({0})'.format(n),
#                         f_nptoman= lambda data: GL3_nptoman(data,n),
#                         f_mantonp= GL_mantonp
#                     )

def get_Affn(n):
    return Manifold(
        n_dimM=n ** 2 + n, n_dimT=n ** 2 + n,
        exp_e=aff_exp_e, log_e=aff_log_e,
        log=aff_log, exp=aff_exp,
        id_elem=np.eye(n + 1), name='Aff({0})'.format(n),
        f_nptoman=lambda data: aff_nptoman(data, n),
        f_mantonp=lambda data: aff_mantonp(data, n)
    )

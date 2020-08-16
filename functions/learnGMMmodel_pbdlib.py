import numpy as np
import riepybdlib.manifold as rm   # For the manifold functionality
import riepybdlib.statistics as rs # For statistics on riemannian manifolds
from riepybdlib.angular_representations import Quaternion
import time


def learnGMMmodel_pbdlib(model, xIn, xOut, nbStates, nbIterEM):
    m_3deucl = rm.get_euclidean_manifold(3, '3d-eucl')
    m_time = rm.get_euclidean_manifold(1, 'time')
    m_quat = rm.get_quaternion_manifold('orientation')

    xIn = np.array([xIn]).T
    x = (xIn, Quaternion.from_nparray(xOut[1].T), Quaternion.from_nparray(xOut[2].T), Quaternion.from_nparray(xOut[3].T)
         , Quaternion.from_nparray(xOut[4].T), Quaternion.from_nparray(xOut[5].T), Quaternion.from_nparray(xOut[6].T)
         , Quaternion.from_nparray(xOut[7].T), Quaternion.from_nparray(xOut[8].T), Quaternion.from_nparray(xOut[9].T),
         xOut[10].T, xOut[11].T, xOut[12].T, xOut[13].T, xOut[14].T, xOut[15].T)

    GMM = rs.GMM(m_time * m_quat * m_quat * m_quat * m_quat * m_quat * m_quat * m_quat * m_quat * m_quat
                 * m_3deucl * m_3deucl * m_3deucl * m_3deucl * m_3deucl * m_3deucl, n_components=nbStates)

    GMM.init_time_based(x[0], x, reg_lambda=model.params_diagRegFact, reg_type=rs.RegularizationType.DIAGONAL)

    t1 = time.perf_counter()
    GMM.fit(x, maxsteps=nbIterEM, reg_lambda=model.params_diagRegFact, reg_type=rs.RegularizationType.DIAGONAL)
    t2 = time.perf_counter()
    print("function fit() takes ", t2 - t1, 's')

    model.Mu = np.zeros((model.nbVar, model.nbStates))
    model.Priors = GMM.priors
    model.Sigma = np.zeros((model.nbVar, model.nbVar, nbStates))
    model.MuMan = np.zeros((model.nbVarMan, nbStates))
    for i, gauss in enumerate(GMM.gaussians):
        model.Sigma[:, :, i] = gauss.sigma
        model.MuMan[:, i] = gauss.manifold.manifold_to_np(gauss.mu)
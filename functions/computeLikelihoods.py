import numpy as np
from functions.mapping import logmap
from functions.Nodes import Lnode
from m_fcts.gaussPDF import gaussPDF


def computeLikelihoods(m_dt, m_nbStates, m_nbVar, m_Priors, m_Mu, m_Sigma, m_MuMan, dataTest, sizeData):
    xIn = np.arange(1, sizeData+1) * m_dt
    L = np.zeros((m_nbStates, sizeData))
    Data_gaussPDF = np.zeros((m_nbVar, sizeData, m_nbStates))
    Lglobal = {}
    Lbodypart = {}
    Ljoints = {}

    # Global(orientation + position)
    for i in range(m_nbStates):
        Data_gaussPDF[0, :, i] = xIn - m_MuMan[0, i]
        Data_gaussPDF[1:4, :, i] = logmap(dataTest['lElbow ori'], m_MuMan[1:5, i])
        Data_gaussPDF[4:7, :, i] = logmap(dataTest['lWrist ori'], m_MuMan[5:9, i])
        Data_gaussPDF[7:10, :, i] = logmap(dataTest['lShoulder ori'], m_MuMan[9:13, i])
        Data_gaussPDF[10:13, :, i] = logmap(dataTest['rElbow ori'], m_MuMan[13:17, i])
        Data_gaussPDF[13:16, :, i] = logmap(dataTest['rWrist ori'], m_MuMan[17:21, i])
        Data_gaussPDF[16:19, :, i] = logmap(dataTest['rShoulder ori'], m_MuMan[21:25, i])
        Data_gaussPDF[19:22, :, i] = logmap(dataTest['mSpine ori'], m_MuMan[25:29, i])
        Data_gaussPDF[22:25, :, i] = logmap(dataTest['mShoulder ori'], m_MuMan[29:33, i])
        Data_gaussPDF[25:28, :, i] = logmap(dataTest['Neck ori'], m_MuMan[33:37, i])
        Data_gaussPDF[28:31, :, i] = dataTest['lElbow rel_pos'] - m_MuMan[37:40, i:i + 1]
        Data_gaussPDF[31:34, :, i] = dataTest['lWrist rel_pos'] - m_MuMan[40:43, i:i + 1]
        Data_gaussPDF[34:37, :, i] = dataTest['lShoulder rel_pos'] - m_MuMan[43:46, i:i + 1]
        Data_gaussPDF[37:40, :, i] = dataTest['rElbow rel_pos'] - m_MuMan[46:49, i:i + 1]
        Data_gaussPDF[40:43, :, i] = dataTest['rWrist rel_pos'] - m_MuMan[49:52, i:i + 1]
        Data_gaussPDF[43:46, :, i] = dataTest['rShoulder rel_pos'] - m_MuMan[52:55, i:i + 1]

        L[i:i + 1, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[:, :, i], m_Mu[:, i], m_Sigma[:, :, i]).T
    LL = np.log(np.sum(L, axis=0))
    LL = np.where(LL < -2000, -2000, LL)
    score = np.mean(LL)
    Lglobal['Global'] = Lnode(LL, score)

    sigma = np.zeros((28, 28))
    # Orientations
    # out = 2:28 omitted to have a faster calulation
    for i in range(m_nbStates):
        mu = m_Mu[0:28, i]
        sigma[0, 0] = m_Sigma[0, 0, i]
        sigma[0, 1:] = m_Sigma[0, 1:28, i]
        sigma[1:, 0] = m_Sigma[1:28, 0, i]
        sigma[1:, 1:] = m_Sigma[1:28, 1:28, i]
        L[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[:28, :, i], mu, sigma)
    LL = np.log(np.sum(L, axis=0))
    LL = np.where(LL < -2000, -2000, LL)
    score = np.mean(LL)
    Lglobal['Orientations'] = Lnode(LL, score)

    # Positions
    # out = 29:46 omitted to have a faster calulation
    sigma = np.zeros((19, 19))
    index = [0, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    for i in range(m_nbStates):
        mu = m_Mu[index, i]
        sigma[0, 0] = m_Sigma[0, 0, i]
        sigma[0, 1:] = m_Sigma[0, 28:46, i]
        sigma[1:, 0] = m_Sigma[28:46, 0, i]
        sigma[1:, 1:] = m_Sigma[28:46, 28:46, i]
        L[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[index, :, i], mu, sigma)
        LL = np.log(np.sum(L, axis=0))
        LL = np.where(LL < -2000, -2000, LL)
        score = np.mean(LL)
        Lglobal['Positions'] = Lnode(LL, score)

    return [], [], []

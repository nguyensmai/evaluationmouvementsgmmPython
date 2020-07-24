import numpy as np


def init_GMM_kbins(Data, model, nbSamples):
    """
    Initialization of Gaussian Mixture Model (GMM) parameters by clustering
    an ordered dataset into equal bins.
    """
    nbData = Data.shape[1] / nbSamples
    if model.params_diagRegFact == None:
        model.params_diagRegFact = 1e-4 #Optional regularization term to avoid numerical instability

    # Delimit the cluster bins for the first demonstration
    tSep = np.round(np.linspace(0, nbData, model.nbStates + 1)).astype(int)  ##round function is not well defined in python

    # Compute statistics for each bin
    model.Priors = np.zeros(model.nbStates)
    model.Mu = np.zeros((Data.shape[0], model.nbStates))
    model.Sigma = np.zeros((Data.shape[0], Data.shape[0], model.nbStates))

    ## i != 0
    for i in range(0, model.nbStates):
        id = np.array([])
        for n in range(nbSamples):
            id = np.hstack((id, n*nbData + np.array(range(tSep[i],tSep[i+1])))).astype(int)
        model.Priors[i] = id.size
        aee = np.mean(Data[:,id], axis=1,keepdims=True)
        model.Mu[:,i:i+1] = np.mean(Data[:,id], axis=1, keepdims=True)
        er = Data[:,id]
        ae = np.cov(Data[:,id])+np.identity(Data.shape[0]) * model.params_diagRegFact

        model.Sigma[:,:,i] = np.cov(Data[:,id]) + np.identity(Data.shape[0]) * model.params_diagRegFact
    model.Priors = model.Priors / np.sum(model.Priors)
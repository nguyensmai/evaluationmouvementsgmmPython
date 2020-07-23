import numpy as np
from m_fcts.init_GMM_kbins import init_GMM_kbins
from functions.mapping import logmap
from functions.mapping import expmap


def learnGMMmodel(model,u,xIn,xOut,nbSamples,nbIterEM,nbIter,nbData):
    model = init_GMM_kbins(u, model, nbSamples)

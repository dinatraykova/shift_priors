#!/usr/bin/python
from montepython.likelihood_class import Likelihood
import numpy as np
from scipy.stats import multivariate_normal

def get_imodel(model):
    d = {'shift_L0': 0,
        'shift_Lvar': 1,
    return d[model]

class shift_priors(Likelihood):
    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.

    def multvar_normal_logpdf(self, x, model):
        if model == 0:
           means = np.array(self.means)
           cov = np.array(self.cov)
        elif model == 1:
           means = np.array(self.meansL)
           cov = np.array(self.covL)
        else:
           print('Shift model specified does not exist; Choose shift_L0 for Lambda=0 and shift_Lvar for Lambda!=0 model')
        return multivariate_normal.logpdf(x,means,cov)

    def loglkl(self, cosmo, data):
        # self.model must be in param file

        aB0 = cosmo.pars['parameters_smg__1']
        m   = cosmo.pars['parameters_smg__2']
        w0  = cosmo.pars['expansion_smg__2']
        wa  = cosmo.pars['expansion_smg__3']

        X1 = aB0
        X2 = m*aB0**(1./6.)
        X3 = w0*m**(1./4.)
        X4 = wa*(m**2.)
         
        X = np.array((X1, X2, X3, X4))

        return multvar_normal_logpdf(X, imodel)

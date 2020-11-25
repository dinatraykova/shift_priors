#!/usr/bin/python
from montepython.likelihood_class import Likelihood
import numpy as np
from scipy.stats import multivariate_normal

def get_imodel(model):
    d = {'shift_L0': 0,
        'shift_Lvar': 1}
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
        print(-1./2.*(x-mu).T.dot(invS).dot(x-mu))
        print(np.log(multivariate_normal.pdf(x,means,cov)/multivariate_normal.pdf(means,means,cov)))
        #return np.log(multivariate_normal.pdf(x,means,cov)/multivariate_normal.pdf(means,means,cov))
        return -1./2.*(x-mu).T.dot(invS).dot(x-mu)

    def loglkl(self, cosmo, data):
        # self.model must be in param file

        aB0 = data.cosmo_arguments['parameters_smg__1']
        m   = data.cosmo_arguments['parameters_smg__2']
        w0  = data.cosmo_arguments['expansion_smg__2']
        wa  = data.cosmo_arguments['expansion_smg__3']

        #print("a", (aB0,m,w0,wa))

        X1 = aB0
        X2 = m*aB0**(1./6.)
        X3 = w0*m**(1./4.)
        X4 = wa*(m**2.)
        #print("x", (X1,X2,X3,X4))
        
        X = np.array((X1, X2, X3, X4))
        imodel = get_imodel(self.model)
        #print(self.multvar_normal_logpdf(X, imodel))

        return self.multvar_normal_logpdf(X, imodel)

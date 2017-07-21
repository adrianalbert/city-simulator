from urban_growth.model import *
from urban_growth.utility import *
from scipy.optimize import minimize

class estimator(settlement_model):

	def __init__(self, **kwargs):
		settlement_model.__init__(self, **kwargs)

	def log_likelihood(self, M1, K, pars, terms = False):
		
		def logbin(k, n, p):
		    return k * np.log(p) + (n - k) * np.log(1 - p) + np.log(special.binom(n, k))

		v_logbin = np.vectorize(logbin)

		p = np.ravel(self.settlement_rate(K, pars))
		n = np.ravel((1 - np.ravel(self.M0)) * self.N_pix)
		k = np.ravel((M1 - self.M0) * self.N_pix)

		lik_terms = v_logbin(k, n, p)
		
		if terms:
			return lik_terms.reshape(self.M.shape)
		else:
			return np.nansum(lik_terms)

	def ll_obj(self, M1, K, pars):
		
		p, grad =   self.settlement_rate(K = K, pars=pars, use_grad = True)
		lls     =   self.N_pix*((M1-self.M0)*np.log(p) + (1-M1)*np.log(1-p))
		grad    =   self.N_pix*((M1-self.M0)/p-(1-M1)/(1-p))*grad

		return np.nanmean(lls), np.nanmean(grad, axis = (1, 2))

	def ml(self, M1, K, pars_0, use_grad = False, opts = {'disp' : False}):
		
		pars_0 = to_vec(pars_0)

		if use_grad:
			def f(pars): 			
				pars = from_vec(pars)
				ll, grad = self.ll_obj(M1, K, pars)			
				return -ll, -grad
		else:
			def f(pars): 			
				pars = from_vec(pars)
				ll, grad = self.ll_obj(M1, K, pars)			
				print  str(-ll) + ' || ' + str(grad)
				return -ll

		res = minimize(f, 
					   pars_0, 
					   method = 'BFGS', 
					   jac = use_grad, # implement eventually
					   options = opts,
					   tol = .0000001)

		pars = from_vec(res.x)
		
		return pars, - res.fun, res.hess_inv

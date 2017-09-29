from urban_growth.model import *
from urban_growth.utility import *
from scipy.optimize import minimize

class estimator(settlement_model):

	def __init__(self, **kwargs):
		settlement_model.__init__(self, **kwargs)

	def log_likelihood(self, M1, K, pars, terms = False):
		
		p       =   self.settlement_rate(K = K, pars=pars, use_grad = False)
		p[self.M0 == 1] = np.nan
		lls           =   (((M1-self.M0)*np.log(p) + (1-M1)*np.log(1-p))) 
		
		return np.nanmean(lls)

	def ll_obj(self, M1, K, pars):
		
		p, grad       =   self.settlement_rate(K = K, pars=pars, use_grad = True)

		p[self.M0 == 1] = np.nan
		# this expression doesn't factor in already maximally settled areas, right?
		lls           =   (((M1-self.M0)*np.log(p) + (1-M1)*np.log(1-p))) 
		grad_coefs    =   ((M1-self.M0)/p-(1-M1)/(1-p)) * (1 - self.M0)

		grad = grad_coefs*grad

		return np.nanmean(lls), np.nanmean(grad, axis = (1, 2))

	def ml(self, M1, K, pars_0, use_grad = False, opts = {'disp' : False}):
		
		pars_0 = to_vec(pars_0)
	
		def f(pars): 			
			pars = from_vec(pars)
			ll, grad = self.ll_obj(M1, K, pars)	
	
			if use_grad:
				return -ll, -grad
			else:
				return -ll
		
		res = minimize(f, 
					   pars_0, 
					   method = 'BFGS', 
					   jac = use_grad, # implement eventually
					   options = opts,
					   tol = .0000001)

		pars = from_vec(res.x)
		
		return pars, - res.fun, res.hess_inv

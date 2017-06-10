from urban_growth.model import *
from urban_growth.utility import *
from urban_growth.components import *
from scipy.optimize import minimize

class estimator(settlement_model):

	def __init__(self, **kwargs):
		settlement_model.__init__(self, **kwargs)

	# first test is maybe computing log-likelihood. 

	def log_likelihood(self, X, normalized = False, use_grad = False, **pars):
		pi = pars['pi']
		pars.pop('pi')

		if use_grad:
			c, grads = models[self.model]['components'](self, use_grad = use_grad, **pars)
		else:
			c = models[self.model]['components'](self, **pars) # components
		
		d = (np.array([[pi]]).T * c).sum(axis = 0)    # density

		ll = np.nansum(X*np.log(d)+(1-X)*np.log(1-d))

		if normalized: 
			ll = ll / np.isfinite(X).sum()

		# THE BELOW IS WRONG FOR NOW, POSSIBLY COME BACK TO THIS. 
		if use_grad:
			
			coef = X / d - (1 - X) / (1 - d)
			
			pi = np.expand_dims(np.expand_dims(np.expand_dims(pi, 0), 2), 3)

			d_theta = grads
			d_pi    = c / pi # componentwise

			grads = np.concatenate((d_theta, d_pi))
			grad  = np.nansum(pi * coef * grads, axis = (2, 3))
			if normalized:
				grad = grad / np.isfinite(X).sum()
			return ll, grad	
		return ll

	def ML(self, X, pars, use_grad = False, opts = {'disp' : False}):
		
		if use_grad: 
			def f(pars): 
				
				pars = from_vec(pars, self.model)
				pars = from_array(pars, self.model)
				
				ll, grad = self.log_likelihood(X, True, True, **pars)
				
				return - ll, - to_vec(grad)
		else:
			def f(pars): 
				
				pars = from_vec(pars, self.model)
				pars = from_array(pars, self.model)
				
				ll = self.log_likelihood(X, True, False, **pars)
				
				return - ll

		pars_0 = to_array(pars, self.model)
		pars_0 = to_vec(pars_0, self.model)
		
		res = minimize(f, 
					   pars_0, 
					   method = 'BFGS', 
					   jac = use_grad, # implement eventually
					   options = opts,
					   tol = .0000001)

		pars = from_vec(res.x, self.model)
		pars = from_array(pars, self.model)
		
		return pars, - res.fun, res.hess_inv
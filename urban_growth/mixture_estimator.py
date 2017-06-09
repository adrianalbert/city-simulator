from urban_growth.estimator_refactor import *
from scipy.optimize import minimize

class mixture_estimator(estimator):

	def __init__(self, **kwargs):
		estimator.__init__(self, **kwargs)

	# first test is maybe computing log-likelihood. 

	def mixture_log_likelihood(self, X, model, normalized = False, use_grad = False, **pars):
		pi = pars['pi']
		pars.pop('pi')

		if use_grad:
			c, grads = models[model]['components'](self, use_grad = use_grad, **pars)	
		else:
			c = models[model]['components'](self, **pars) # components
		
		d = (np.array([[pi]]).T * c).sum(axis = 0)    # density

		ll = np.nansum(X*np.log(d)+(1-X)*np.log(1-d))

		if normalized: 
			ll = ll / np.isfinite(X).sum()

		# THE BELOW IS WRONG FOR NOW, POSSIBLY COME BACK TO THIS. 
		# if use_grad:
		# 	pi = np.expand_dims(np.expand_dims(np.expand_dims(pi, 0), 2), 3)
		# 	coef = X / d + (1 - X) / (1 - d)
		# 	component_grad = np.nansum(pi * coef * grads, axis = (2, 3))
		# 	d_pi = np.nansum(coef * c, axis = (1, 2))
		# 	d_pi = np.expand_dims(d_pi, 0) 
		# 	print component_grad.shape, d_pi.shape
		# 	grad = np.concatenate((component_grad, d_pi))

		# 	if normalized:
		# 		grad = grad / np.isfinite(X).sum()
		# 	return ll, grad	
		return ll

	def EM(self, M1, model, pars_0, n_iters, use_grad = False, decoupled = False, verbose = True, print_every = 10, tol = .0001):
		X = M1 - self.M0       # only the new settlements
		pars_hat = pars_0
		
		lik = self.log_likelihood(M1, 
								  model, 
								  normalized = True, 
								  **pars_hat)
		
		print '0 : ll = ' + str(np.round(lik, 8))

		for j in range(n_iters):
			self.E_step(X, model = model, **pars_hat)
			pars_hat = self.M_step(X, model, use_grad, decoupled,  **pars_hat)
			if verbose & ((j + 1) % print_every == 0):
				old_lik = lik
				lik = self.log_likelihood(M1,
										  model = model, 
										  normalized = True, 
										  **pars_hat)
				print str(j + 1) + ' : ll = ' + str(np.round(lik, 8)) 
				if lik - old_lik < tol: 
					return pars_hat, lik
		
		return pars_hat, lik

	def E_step(self, X, model, **pars):
		'''
		Compute current estimate under the parameters of belonging to a particular settlement type
		'''
		pi = np.array([[pars['pi']]]).T
		pars.pop('pi')

		# should depend on the data!! 
		D = models[model]['components'](self, **pars)
		terms = pi*(D ** X)*(1 - D)**(1-X)
		self.Q = terms/terms.sum(axis=0, keepdims=True) 

	def M_step(self, X, model, use_grad, decoupled = False, **pars):

		def from_array_2(arr):	
			d = {'alpha' : arr[0], 'gamma' : arr[-1]}
			if arr.shape[0] > 2:
				d.update({'pi' : arr[1]})
			return d

		pi_hat = np.nanmean(self.Q, axis = (1, 2))

		pars.pop('pi')

		n = len(pars.values()[0])
		m = len(pars)
		def f(pars, component = None):
			if component is None:
				pars = np.reshape(pars, (n, m))
				Q = self.Q
			else:
				Q = self.Q[component]

			pars = from_array_2(pars)

			if use_grad:
				c, grads = models[model]['components'](self, use_grad = use_grad, component = component, **pars)	
				coef = Q * (X / c - (1 - X) / (1 - c))
				
				coef = np.expand_dims(coef, axis = 0)
				grad = np.nansum(coef * grads, axis = (1, 2)) / np.isfinite(X).sum()
				
			else:
				c = models[model]['components'](self, use_grad = use_grad, component = component, **pars)	

			ll_comps = X * np.log(c) + (1 - X) * np.log(1 - c)
			E_ll = np.nansum(Q * ll_comps) / np.isfinite(X).sum()

			if use_grad:			
				return - E_ll, - grad
			else:
				return - E_ll

		if not decoupled: 
			pars_0 = np.reshape(to_array(**pars), 4)
			
			res = minimize(f, 
						   pars_0, 
						   method = 'BFGS', 
						   jac = use_grad, 
						   options = {'disp' : True})
			pars = np.reshape(res.x, (2, 2))
		else: 
			def h(k):
				
				pars_0_comp = to_array(**pars)[:,k].T
				
				res = minimize(f, 
				               pars_0_comp,
				               args = (k), 
						   	   method = 'BFGS', 
						       jac = use_grad, 
						       options = {'disp' : True})
				return res.x

			pars = np.array([h(k) for k in range(self.types.shape[0])]).T

		pars = from_array_2(pars)
		pars['pi'] = pi_hat
		return pars

	def ML(self, X, model, opts = {'eps' : .0000001, 'disp' : False},  **pars):
		
		def from_array_2(arr):	
			d = {'alpha' : arr[0], 'gamma' : arr[-1]}
			if arr.shape[0] > 2:
				d.update({'pi' : arr[1]})
			return d
		
		shape = len(pars), len(pars[pars.keys()[0]])

		def f(pars): 
			
			pars = np.reshape(pars, shape)
			pars = from_array_2(pars)
			
			ll = self.mixture_log_likelihood(X, 
			                                     model, 
			                                     normalized = True, 
			                                     use_grad = False, 
			                                     **pars)
			
			return - ll

		pars_0 = to_array(**pars)
		pars_0 = pars_0.reshape(shape[0] * shape[1])
		res = minimize(f, 
					   pars_0, 
					   method = 'BFGS', 
					   jac = False, 
					   options = opts,
					   tol = .0000001)

		pars = np.reshape(res.x, shape)
		pars = from_array_2(pars)
		
		return pars, - res.fun, res.hess_inv









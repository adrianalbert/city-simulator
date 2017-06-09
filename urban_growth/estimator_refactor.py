from urban_growth.model_refactor import *
from scipy.optimize import minimize

class estimator(settlement_model):

	def __init__(self, **kwargs):
		settlement_model.__init__(self, **kwargs)

	def log_likelihood(self, M1, model, normalized=False,  **pars):
		'''
		In this implementation, need to have partition clusters and distance features called first. 
		'''
		X = M1 - self.M0
		X[self.get_M0() == 1] = np.nan

		d = models[model]['density'](self, **pars)
		

		ll = np.nansum(X*np.log(d)+(1-X)*np.log(1-d))
		if normalized:
			ll = 1.0 * ll / np.isfinite(X).sum()
		return ll

	def EM(self, M1, model, pars_0, n_iters, verbose = True, print_every = 10, tol = .01):
		X = M1 - self.M0       # only the new settlements
		pars_hat = pars_0
		
		lik = self.log_likelihood(M1, 
		                          model, 
		                          normalized = True, 
		                          **pars_hat)
		
		print 'll0 : ' + str(np.round(lik, 2))

		for j in range(n_iters):
			self.E_step(model = model, **pars_hat)
			pars_hat = self.packaged_M_step(X, model, pars_hat)
			# pars_hat = self.M_step(X, model, pars_hat)
			if verbose & (j % print_every == 0):
				old_lik = lik
				lik = self.log_likelihood(M1,
										  model = model, 
				                          normalized = True, 
				                          **pars_hat)
				print str(j) + ': ll  = ' + str(np.round(lik, 2)) 
				if lik - old_lik < tol: 
					return pars_hat, lik
		
		return pars_hat, lik

	def E_step(self, model, **pars):
		'''
		Compute current estimate under the parameters of belonging to a particular settlement type
		'''
		D = models[model]['components'](self, **pars)
		self.Q = D / D.sum(axis=0, keepdims=True) 

	def M_step(self, X, model, pars):
		
		# this is where the gradient step updates will be. 
		# need a way to efficiently handle flexible parameters. 
		# shoving everything into an array and extracting on the fly might be the way. 

		for i in range(n_inner):
			grad = self.expected_gradient(X, model, **pars)			
			pars = to_array(**pars)
			pars += eta * grad
			pars = from_array(pars)

			# if inner tolerance is met, return current estimates
			if eta ** 2 * np.sum(grad ** 2) / np.size(grad) < inner_tol:
				return pars

		# if we have run through n_inner, return current estimates.
		return pars

	# consider doing this as basin-hopping
	def packaged_M_step(self, X, model, pars):
		
		def f(pars): 
			
			pars = np.reshape(pars, (3, 2))
			pars = from_array(pars)
			
			# compute expected log likelihood
			c = models[model]['components'](self, **pars)
			q = c / np.nansum(c, keepdims=True)

			ll_comps = np.log(q) + X * np.log(c) + (1 - X) * np.log(1 - c)
			E_ll = np.nansum(self.Q * ll_comps) / np.isfinite(X).sum()
			
			# compute expected gradient (seek way to share info between these two steps)
			grad = gradient(self, X, model, **pars)
			E_grad = np.nansum(np.array([self.Q]) * grad, axis = (2, 3)) / np.nansum(X)

			E_grad = np.reshape(E_grad, 6)
			# print E_ll, E_grad

			# return - E_ll, - E_grad
			return -E_ll

		pars_0 = np.reshape(to_array(**pars), 6)
		res = minimize(f, 
		               pars_0, 
		               method = 'BFGS', 
		               jac = False, 
		               options = {'eps' : .000001, 'disp' : True},
		               tol = .0000001)

		pars = np.reshape(res.x, (3, 2))
		pars = from_array(pars)
		
		return pars

	def expected_gradient(self, X, model, **pars):
		'''
		See notes for formula
		'''
		
		grad = gradient(self, X, model, **pars)
		expected_grad = np.nansum(np.array([self.Q]) * grad, axis = (2, 3)) / np.nansum(X)
		return expected_grad

	def expected_ll(self, X, model, **pars):
		c = models[model]['components'](self, **pars)
		q = c / np.nansum(c, keepdims=True)

		ll_comps = q + X * np.log(c) + (1 - X) * np.log(1 - c)

		return np.nansum(self.Q * ll_comps) / np.isfinite(X).sum()

	def ML(self, X, model, pars):
		shape = len(pars), len(pars[pars.keys()[0]])
		def f(pars): 
			
			pars = np.reshape(pars, shape)
			pars = from_array(pars)
			
			return - self.log_likelihood(self.M0 + X, model, normalized = True, **pars)

		def grad_func(pars):
			# doesn't work yet
			
			pars = np.reshape(pars, shape)
			pars = from_array(pars)

			e_grad = - applyself.expected_gradient(X, model, **pars)
			return np.reshape(e_grad, shape[0] * shape[1])

		pars_0 = to_array(**pars)
		res = minimize(f, 
		               pars_0, 
		               method = 'BFGS', 
		               # jac = grad_func, 
		               options = {'eps' : .000001},
		               tol = .0000001)

		pars = np.reshape(res.x, shape)
		pars = from_array(pars)
		
		return pars, res.fun, res.hess_inv





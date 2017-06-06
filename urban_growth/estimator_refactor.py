from urban_growth.model_refactor import *

class estimator(settlement_model):

	def __init__(self, **kwargs):
		settlement_model.__init__(self, **kwargs)

	def log_likelihood(self, M1, model, normalized=False, **pars):
		'''
		In this implementation, need to have partition clusters and distance features called first. 
		'''
		X = M1 - self.M0
		X[self.get_M0() == 1] = np.nan

		d = models[model]['density'](self, **pars)

		ll = np.nansum(X*np.log(d)+(1-X)*np.log(1-d))
		if normalized:
			ll = 1.0 * ll / np.nansum(X)
		return ll

	def EM(self, M1, model, pars_0, n_iters, verbose = True, print_every = 10, tol = .01, **inner_args):
		X = M1 - self.M0       # only the new settlements
		pars_hat = pars_0
		
		lik = self.log_likelihood(M1, 
		                          model, 
		                          normalized = True, 
		                          **pars_hat)
		
		print 'll0 : ' + str(np.round(lik, 2))

		for j in range(n_iters):
			self.E_step(model = model, **pars_hat)
			pars_hat = self.M_step(X, model, pars_hat, **inner_args)
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

	def M_step(self, X, model, pars, eta, n_inner, inner_tol = .1):
		
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

	def expected_gradient(self, X, model, **pars):
		'''
		See notes for formula
		'''
		
		grad = gradient(self, X, model, **pars)
		expected_grad = np.nansum(np.array([self.Q]) * grad, axis = (2, 3)) / np.nansum(X)
		return expected_grad




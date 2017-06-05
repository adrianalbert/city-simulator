from urban_growth.model_refactor import *

class estimator(settlement_model):

	def __init__(self, **kwargs):
		settlement_model.__init__(self, **kwargs)

	def log_likelihood(self,M1,normalized=False,**kwargs):
		'''
		In this implementation, need to have partition clusters and distance features called first. 
		'''
		X = M1 - self.M0
		X[self.get_M0() == 1] = np.nan

		d = self.logistic_density(**kwargs)

		ll = np.nansum(X*np.log(d)+(1-X)*np.log(1-d))
		if normalized:
			ll = 1.0 * ll / np.nansum(X)
		return ll

	def EM(self, M1, alpha_0, beta_0, gamma_0, n_iters, verbose = True, tol = .01, **inner_args):
		X = M1 - self.M0       # only the new settlements
		alpha_hat, beta_hat, gamma_hat = alpha_0, beta_0, gamma_0
		
		lik = self.log_likelihood(M1, 
		                          normalized = True, 
		                          alpha = alpha_hat, 
		                          beta = beta_hat, 
		                          gamma = gamma_hat)
		
		print 'll0 : ' + str(np.round(lik, 2))

		for j in range(n_iters):
			self.E_step(alpha = alpha_hat, beta = beta_hat, gamma = gamma_hat)
			alpha_hat, beta_hat, gamma_hat = self.M_step(X, alpha_hat, beta_hat, gamma_hat, **inner_args)
			if verbose:
				old_lik = lik
				lik = self.log_likelihood(M1, 
				                          normalized = True, 
				                          alpha = alpha_hat, 
				                          beta = beta_hat, 
				                          gamma = gamma_hat)
				print 'll  : ' + str(np.round(lik, 2)) 
				if lik - old_lik < tol: 
					return alpha_hat, beta_hat, gamma_hat, lik
		
		return alpha_hat, beta_hat, gamma_hat, lik

	def E_step(self, **kwargs):
		'''
		Compute current estimate under the parameters of belonging to a particular settlement type
		'''
		D = self.logistic_components(**kwargs)
		self.Q = D / D.sum(axis=0, keepdims=True) 

	def M_step(self, X, alpha, beta, gamma, eta, n_inner, inner_tol = .1):
		
		alpha_hat, beta_hat, gamma_hat = alpha, beta, gamma
		for i in range(n_inner):
			grad = self.gradient(X, alpha_hat, beta_hat, gamma_hat)
			
			alpha_hat += eta * grad[0]
			beta_hat  += eta * grad[1]
			gamma_hat += eta * grad[2]

			# if inner tolerance is met, return current estimates
			if eta ** 2 * np.sum(grad ** 2) / np.size(grad) < inner_tol:
				return alpha_hat, beta_hat, gamma_hat

		# if we have run through n_inner, return current estimates.
		return alpha_hat, beta_hat, gamma_hat

	def gradient(self, X, alpha, beta, gamma):
		'''
		See notes for formula
		'''
		p = self.logistic_components(alpha = alpha, beta = beta, gamma = gamma)

		alpha_arr = np.array([[alpha]]).T
		beta_arr = np.array([[beta]]).T
		gamma_arr = np.array([[gamma]]).T

		coef = X / p - (1 - X) / (1 - p)

		# alpha
		f_gamma = self.distance_feature(np.array([[np.dot(gamma, self.types)]]).T, partitioned = True) 
		d_alpha = p * (1 - p) * f_gamma

		# beta
		d_beta =  p * (1 - p)

		# gamma
		f_gamma_1 = self.distance_feature(np.array([[np.dot(gamma + 1, self.types)]]).T, partitioned = True) 
		d_gamma   = - alpha_arr * gamma_arr * p * (1 - p) * f_gamma_1

		dp = np.array([d_alpha, d_beta, d_gamma])

		dq = dp / np.array(p) - np.nansum(dp, axis = 1, keepdims=True) / np.nansum(p, axis = 0, keepdims=True)
		out = dq + coef * dp

		grad = np.nansum(np.array([self.Q]) * out, axis = (2, 3)) / np.nansum(X)

		return grad




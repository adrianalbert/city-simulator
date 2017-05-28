from urban_growth.model import *

class estimator(settlement_model):

	def __init__(self, **kwargs):
		
		settlement_model.__init__(self, **kwargs)
		self.Q = None
		self.estimate = None

	def log_likelihood(self, M1, **kwargs):
		'''
		General for supervised models with distance weights and latent urban classes
		'''
		
		probs = self.density(**kwargs)
		prob = probs[0] + probs[1]
		
		X = M1 - self.M0

		ll = np.nansum(X * np.log(prob) + (1 - X) * np.log(1 - prob))
		
		if normalized:
			N = (1 - np.isnan(M1)).sum()
			return 1.0 / N * ll
		
		else:
			return ll
	def EM(self, M1, n_iters = None, eta = .005, tol = 1e-3, **kwargs):

		done = False
		while not done:
			# do some computations
			d_est = self.estimate - new_estimate
			converged = np.dot(d_est,d_est)  < tol 
			exceeded_iters = (n_iters is not None) & (i >= n_iters)
			done = converged | exceeded_iters

	# core inference functions
	def M_step(self,  **kwargs):
		'''
		Q is the latent probability of being "really" an urban cell. 
		It is computed as a responsibility in Gaussian mixture modeling.
		'''
		
		probs = self.density(pars = self.estimate, **kwargs)
		self.Q = 1.0 * p[1] / (p[0] + p[1])
		
	def E_step(self, method = 'SGD', n_iters = None, eta = .005, tol = 1e-3, **kwargs):
		# consider whether we even need to bother implementing GD
		if method == 'SGD':
			done = False
			while not done:
				# pick a random point i from support(M1 - M0).
				# grad = SGD(i, model, estimate)
				
				converged = np.dot(grad, grad)  < tol 
				exceeded_iters = (n_iters is not None) & (i >= n_iters)
				done = converged | exceeded_iters

			self.estimate = new_estimate

		print 'not implemented'
		print 'log likelihood : ' + self.log_likelihood(M1 ,**kwargs)

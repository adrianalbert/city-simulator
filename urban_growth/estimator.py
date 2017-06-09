from urban_growth.model import *

class estimator(settlement_model):

	def __init__(self, **kwargs):
		
		settlement_model.__init__(self, **kwargs)
		self.Q = None
		self.estimate = None


	def log_likelihood(self, M1, normalized = True, **kwargs):
		'''
		General for supervised models with distance weights and latent urban classes
		'''
		
		probs = self.density(**kwargs)
		prob = probs[0] + probs[1]

		X = M1 - self.M0

		ll = np.nansum(X * np.log(prob) + (1 - X) * np.log(1 - prob))
		
		if normalized:
			N = M1.size - self.M0.sum()
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

	def initial_estimate(self, initial_pars):
		self.estimate = initial_pars

	# core inference functions
	def E_step(self,  **kwargs):
		'''
		Q is the latent probability of being "really" an urban cell. 
		It is computed as a responsibility in Gaussian mixture modeling.
		'''
		
		p = self.density(pars = self.estimate, **kwargs)
		self.Q = 1.0 * p[1] / (p[0] + p[1])
		
	def M_step(self, M1, method = 'SGD', n_iters = None, eta = .005, tol = 1e-3, print_every = None, **kwargs): # question: do we have singularities here associated with large gamma_u?
		# consider whether we even need to bother implementing GD
		par_vec = to_vec(self.estimate)
		if method == 'SGD':
			done = False
			i = 0
			while not done:
				
				# do computation
				
				grad = self.SG(M1, par_vec, model = model_0)
				par_vec = par_vec + eta * grad
				if (print_every is not None) and (i % print_every == 0):
					print par_vec
				i += 1
				
				# check if we're done
				converged = np.dot(grad, grad)  < tol 
				exceeded_iters = (n_iters is not None) & (i >= n_iters)
				done = converged | exceeded_iters

			self.estimate = from_vec(par_vec)

		print 'log likelihood : ' + str(self.log_likelihood(M1, stage = 'initial', pars = self.estimate, truncation = None, **kwargs))

	def SG(self, M1, par_vec, model = model_0):
		'''
			Needs to consider geo? A calculation suggests no, but that seems as though it can't be right. 
		'''
		# seems to increase regardless of the value of x sampled

		pars = from_vec(par_vec)

		# random data point
		i = np.random.choice(self.dists[0].keys()) 
		ix = self.unsettled[i]

		# matrix entries
		x   = M1[ix[0], ix[1]]         
		q   = self.Q[ix[0], ix[1]]     # now have probabilities and q, need gradients
		
		# need single probability density at data point i. 
		grad_r, grad_u = self.model_0_grad(i, par_vec)

		dist_r = self.dists[0][i]
		dist_u = self.dists[1][i]

		w_r = np.dot(dist_r[:,0] ** (-pars['gamma_r']), dist_r[:,1])
		w_u = np.dot(dist_u[:,0] ** (-pars['gamma_u']), dist_u[:,1])

		p_r, p_u = model_0(w_r, w_u, **pars)

		first_term_r  = grad_r / p_r - (grad_r + grad_u) / (p_r + p_u)
		second_term_r = (1.0 * x / p_r - (1.0 - x) / (1.0 - p_r)) * grad_r
		first_term_u  = grad_u / p_u - (grad_r + grad_u) / (p_r + p_u)
		second_term_u = (1.0 * x / p_u - (1.0 - x) / (1.0 - p_u)) * grad_u

		grad = (1 - q) * (first_term_r + second_term_r) + q * (first_term_u + second_term_u)

		return grad
		
		# now just need to plug in exactly where it should go into the entire expression, resulting
		# which gives us the SG step. Then enclose in a loop and we are good to go. 

	def model_0_grad(self, i, par_vec):
		
		pars = from_vec(par_vec, model = 'model_0')

		# retrieved distances
		dist_r = self.dists[0][i]
		dist_u = self.dists[1][i]

		# weights
		
		w_r = np.dot(dist_r[:,0] ** (-pars['gamma_r']), dist_r[:,1])
		w_u = np.dot(dist_u[:,0] ** (-pars['gamma_u']), dist_u[:,1])

		dw_r = np.dot(dist_r[:,0] ** (-(pars['gamma_r'] + 1)), dist_r[:,1])
		dw_u = np.dot(dist_u[:,0] ** (-(pars['gamma_u'] + 1)), dist_u[:,1])

		# probabilities

		p = model_0(w_r, w_u, **pars)

		d_alpha_r = 1.0 / pars['alpha_r']
		d_alpha_u = 1.0 / pars['alpha_u']

		def d_gamma(p, alpha, gamma, dw):
			return gamma / normalizer(gamma) * (normalizer(gamma + 1.0) * p - alpha * dw)

		d_gamma_r = d_gamma(p[0], pars['alpha_r'], pars['gamma_r'], dw_r)
		d_gamma_u = d_gamma(p[1], pars['alpha_u'], pars['gamma_u'], dw_u)
	
		grad_r = {'alpha_r' : d_alpha_r,
		          'alpha_u' : 0,
		          'gamma_r' : d_gamma_r,
		          'gamma_u' : 0}

		grad_u = {'alpha_r' : 0,
		          'alpha_u' : d_alpha_u,
		          'gamma_r' : 0,
		          'gamma_u' : d_gamma_u}

		return to_vec(grad_r), to_vec(grad_u)



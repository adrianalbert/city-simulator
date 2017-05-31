from urban_growth.model import *
from scipy.spatial import distance

class mixture_model:

	def __init__(self, M0 = None, geo = None):
		if geo is not None:
			self.geo = geo
		if M0 is not None:
			self.set_M0(M0 = M0)

	def set_M0(self, M0 = None, **kwargs):
		if M0 is not None:
			self.M0 = M0
		else:
			self.M0 = random_mat(**kwargs)

		self.M = self.M0.copy()
		
		mask = id_clusters(self.M0)
		self.clusters = {i : np.array(np.where(mask == i)).T for i in np.unique(mask) if i > 0}

		a = np.where(self.M0 == 0)
		self.unsettled = np.array([[a[0][i], a[1][i]] for i in range(len(a[0]))])

	def get_M0(self):
		return self.M0

	def get_M(self):
		return self.M

	def make_dists(self):
		a = np.where(self.M == 0)
		self.unsettled = np.array([[a[0][i], a[1][i]] for i in range(len(a[0]))])

		mask = id_clusters(self.M)
		self.clusters = {i : np.array(np.where(mask == i)).T for i in np.unique(mask) if i > 0}
		
		self.dists = {i : distance.cdist(self.unsettled, 
										 self.clusters[i], 
										 'euclidean') 
					  for i in self.clusters}

	def density(self, alpha, gamma):
		'''
		alpha and gamma are each nonnegative vectors the size of self.clusters.keys().
		'''
		p = self.mixture_components(gamma, normalized = True)
		p = p * np.array([[alpha]]).T

		d = np.sum(p, axis = 0)
		d = d / np.nansum(d)
		return d

	def mixture_components(self, gamma, normalized = True):
		def f(j):
		    vec = sum((self.dists[j][:,i] ** (-gamma[j - 1])) for i in range(self.dists[j].shape[1])) 
		    mat = np.empty(self.M0.shape)
		    mat[:] = np.NAN
		    for i in range(self.unsettled.shape[0]):
		        mat[tuple(self.unsettled[i])] = vec[i] 
		    # adjusted = mat * self.geo
		    if normalized:
		    	mat = mat / np.nansum(mat)
		    return mat

		Q = np.array([f(j) for j in range(1,len(self.dists) + 1)])

		return Q

	def sample(self, n_samples, **kwargs):
		d = self.density(**kwargs)
		d[np.isnan(d)] = 0
		p = np.ravel(d)
		out = np.random.choice(np.arange(len(p)), size = n_samples, p = p)
		self.M[np.unravel_index(out, d.shape)] = 1
		return self.M

	def dynamics(self, n_iters, n_samples):
		M = np.zeros(self.M0.shape)
		for i in range(n_iters):
			self.make_dists()
			self.set_params()
			M += self.sample(n_samples, alpha = self.alpha, gamma = self.gamma)

		return M

	def set_params(self):
		areas = np.array([self.clusters[i + 1].shape[0] for i in range(len(self.clusters))])
		self.alpha = areas
		self.gamma = 2 + 5.0 / areas

	def log_likelihood(self, M1, normalized = False, **pars):
		d = self.density(**pars)
		X = M1 - self.M0
		lik = np.nansum((np.log(d) * X))
		if normalized: 
			lik = lik / X.sum()
		return lik

	def EM(self, M1, alpha_0, gamma_0, n_iters, verbose = True, tol = .01, **inner_args):
		X = M1 - self.M0       # only the new settlements
		alpha_hat, gamma_hat = alpha_0, gamma_0
		lik = self.log_likelihood(M1, normalized = True, alpha = alpha_hat, gamma = gamma_hat)
		print 'll0 : ' + str(np.round(lik, 2))

		for j in range(n_iters):
			self.E_step(alpha_hat, gamma_hat)
			alpha_hat, gamma_hat = self.M_step(X, alpha_hat, gamma_hat, **inner_args)
			if verbose:
				old_lik = lik
				lik = self.log_likelihood(M1, normalized = True, alpha = alpha_hat, gamma = gamma_hat)
				print 'll  : ' + str(np.round(lik, 2))
				if lik - old_lik < tol: 
					return alpha_hat, gamma_hat, lik
		
		return alpha_hat, gamma_hat, lik
	
	def E_step(self, alpha, gamma):
		
		p = self.mixture_components(gamma, normalized = True)
		weighted = p * np.array([[alpha]]).T
		self.Q = weighted / weighted.sum(axis = 0)

	def M_step(self, X, alpha, gamma, eta, n_inner, inner_tol = .1):
		
		new_alpha = np.nansum(X / X.sum() * self.Q, axis = (1,2))

		for i in range(n_inner):
			grad = self.gradient(X, gamma)
			new_gamma = gamma + eta * grad
			if 1.0 / len(gamma) * np.sqrt(np.sum((eta * grad)**2)) < inner_tol:
				return new_alpha, new_gamma

		return new_alpha, new_gamma
		

	def gradient(self, X, gamma):
		'''
			No gradients for alpha, since these are found in closed form in the M step
		'''

		f_gamma   = self.mixture_components(gamma,     normalized = False)
		f_gamma_1 = self.mixture_components(gamma + 1, normalized = False)
		
		Z_gamma   = np.nansum(f_gamma,   axis = (1,2))
		Z_gamma_1 = np.nansum(f_gamma_1, axis = (1,2))

		first_term  = Z_gamma_1 / Z_gamma * np.nansum(X * self.Q, axis = (1, 2))
		second_term = np.nansum(X * self.Q * f_gamma_1 / f_gamma, axis = (1,2))

		return gamma * (first_term - second_term)
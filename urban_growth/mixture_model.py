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

	def density(self, alpha, gamma, components = False):
		'''
		alpha and gamma are each nonnegative vectors the size of self.clusters.keys().
		'''
		def f(j):
		    vec = sum((self.dists[j][:,i] ** (-gamma[j - 1])) for i in range(self.dists[j].shape[1])) 
		    mat = np.empty(self.M0.shape)
		    mat[:] = np.NAN
		    for i in range(self.unsettled.shape[0]):
		        mat[tuple(self.unsettled[i])] = vec[i] 
		    adjusted = mat * self.geo
		    adjusted = adjusted / np.nansum(adjusted)
		    return adjusted * alpha[j - 1]

		Q = np.array([f(j) for j in range(1,len(self.dists) + 1)])
		Q = Q / np.nansum(Q)

		if components:
			return Q
		else:
			return np.sum(Q, axis = 0)

		return out


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

	def E_step(self, **pars):
		p = self.density(components = True, **pars)
		self.Q = p / p.sum(axis = 0)











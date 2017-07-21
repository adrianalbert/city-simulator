from urban_growth.model import *
from urban_growth.components import *

class simulator(settlement_model):

	def __init__(self, **kwargs):

		settlement_model.__init__(self, **kwargs)

	
	def sample(self, K, pars):
		v_bin = np.vectorize(np.random.binomial)

		p = np.ravel(self.settlement_rate(K, pars))
		n = np.ravel((1 - np.ravel(self.M)) * self.N_pix)

		samp = 1.0 * v_bin(n, p) / self.N_pix

		return np.reshape(samp, self.M.shape)

	def dynamics(self, K, pars, n_iters = 1):
		
		for i in range(n_iters):
			new = self.sample(K, pars)
			self.M += new
			self.partition_clusters() # update morphology

		return self.M 
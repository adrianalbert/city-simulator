from urban_growth.model import *
from urban_growth.components import *

class simulator(settlement_model):

	def __init__(self, **kwargs):

		settlement_model.__init__(self, **kwargs)

	def sample(self, K, pars):
		s = self.settlement_rate(K, pars, use_grad = False)
		rands = np.random.rand(*s.shape)
		new_mat = (rands < s) * 1
		return new_mat
		
	def dynamics(self, K, pars, n_iters = 1):
		
		for i in range(n_iters):
			new = self.sample(K, pars)
			self.M += new
			self.partition_clusters() # update morphology

		return self.M 
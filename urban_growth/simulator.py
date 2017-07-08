from urban_growth.model import *
from urban_growth.components import *

class simulator(settlement_model):

	def __init__(self, **kwargs):

		settlement_model.__init__(self, **kwargs)

	def sample(self, **pars):
		'''
		Forward step
		'''
		prob = density(self, self.model, **pars)
		prob = prob * self.geo
		rands = np.random.rand(*prob.shape)
		new_mat = (rands < prob) * 1
		return new_mat

	def dynamics(self, T_vec, n_iters = 5, verbose = True, **pars):
		
		times = np.arange(2, n_iters + 2)
		
		for i in times:	
			
			if (self.M == 0).sum() == 0: # if matrix is full, no point in further iterations
				break

			self.update_morphology() # might want to move this somewhere else
			self.make_dist_array()
			self.partition_clusters(T_vec)
			self.partition_dist_array()
			
			s = self.sample(**pars)
			s[self.M > 0] = 1
			
			self.M  += s
						
			if verbose:
				print 'Step ' + str(i - 1) + ' completed'

		self.M = i + 1 - self.M
		self.M[self.M == i + 1] = 0
		return self.M
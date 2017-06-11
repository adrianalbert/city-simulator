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

	def dynamics(self, T_vec, n_iters = 5, verbose = True,  trunc = 50, **pars):
		
		times = np.arange(2, n_iters + 2)
		
		for i in times:	
			
			self.partition_clusters(T_vec)
			self.update_morphology() # might want to move this somewhere else
			self.make_dist_array(trunc = trunc)
			s = self.sample(**pars)
			
			self.M  += s
			
			if verbose:
				print 'Step ' + str(i - 1) + ' completed'
	
		return self.M
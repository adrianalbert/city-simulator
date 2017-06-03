from urban_growth.model_refactor import *

class simulator(settlement_model):

	def __init__(self, **kwargs):

		settlement_model.__init__(self, **kwargs)

	def sample(self, **kwargs):
		'''
			Forward step
		'''

		prob = self.logistic_density(**kwargs)
		rands = np.random.rand(*prob.shape)
		new_mat = (rands < prob) * 1

		return new_mat

	def dynamics(self, T_vec, n_iters = 5, verbose = True, **kwargs):
		'''
			Defaults to model_1 for now
		'''
		times = np.arange(2, n_iters + 2)
		return_mat = self.M.copy()
		for i in times:
			self.update_morphology() # might want to move this somewhere else
			self.edt()
			self.partition_clusters(T_vec)
			self.distance_variables()

			s = self.sample(**kwargs)
			self.M += s
			return_mat += i * s
			if verbose:
				print 'Step ' + str(i - 1) + ' completed'

		return_mat[return_mat == 0] = np.nan 	
		return_mat -= 1
		
		return return_mat
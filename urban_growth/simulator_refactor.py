from urban_growth.model_refactor import *

class simulator(settlement_model):

	def __init__(self, **kwargs):

		settlement_model.__init__(self, **kwargs)

	def sample(self, model, dist_approx,  **pars):
		'''
			Forward step
		'''

		prob = models[model]['density'](self, dist_approx = dist_approx,  **pars)
		rands = np.random.rand(*prob.shape)
		new_mat = (rands < prob) * 1

		return new_mat

	def dynamics(self, model, T_vec, n_iters = 5, verbose = True, return_type = 'plain', trunc = 50, dist_approx = False, **pars):
		'''
			Defaults to model_1 for now
		'''
		times = np.arange(2, n_iters + 2)
		return_mat = self.M.copy()
		for i in times:
			self.partition_clusters(T_vec)
			self.update_morphology() # might want to move this somewhere else
			if dist_approx:
				self.distance_variables()
			else:
				self.make_dist_array(trunc = trunc)

			s = self.sample(model, dist_approx, **pars)
			self.M += s
			return_mat += i * s
			if verbose:
				print 'Step ' + str(i - 1) + ' completed'

		if return_type == 'plain':
			return self.M

		return_mat[return_mat == 0] = np.nan 	
		return_mat -= 1
		
		return return_mat
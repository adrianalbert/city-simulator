from urban_growth.model import *

class simulator(settlement_model):

	def __init__(self, **kwargs):

		settlement_model.__init__(self, **kwargs)

	def sample(self, thresh, pars, use_geo = False, stage = 'current', mode = 'full', truncation = None):
		'''
			Forward step
		'''

		probs = self.density(thresh, pars, use_geo, stage, mode, truncation)
		prob = probs[0] + probs[1]
		rands = np.random.rand(*prob.shape)
		new_mat = (rands < prob) * 1

		return new_mat

	def dynamics(self, thresh, pars, use_geo = False, n_iters = 5, mode = 'full', truncation = None):
			
		times = np.arange(2, n_iters + 2)
		return_mat = self.M.copy()
		for i in times:
			s = self.sample(thresh, pars, use_geo, 'current', mode, truncation)
			self.M += s
			return_mat += i * s

		return_mat[return_mat == 0] = np.nan 	
		return_mat -= 1
		
		return return_mat
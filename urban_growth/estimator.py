from urban_growth.model import *

class estimator(settlement_model):

	def __init__(self, **kwargs):
		
		settlement_model.__init__(self, **kwargs)

	def log_likelihood(self, M1, thresh, pars, model_type, normalized = False, use_geo = False, truncation = None):
		'''
			Correct for both of the supervised models
		'''
		
		probs = self.density(thresh, pars, use_geo, stage = 'initial', truncation = truncation)
		prob = probs[0] + probs[1]
		
		X = M1 - self.M0

		ll = np.nansum(X * np.log(prob) + (1 - X) * np.log(1 - prob))
		
		if normalized:
			N = (1 - np.isnan(M1)).sum()
			return 1.0 / N * ll
		
		else:
			return ll
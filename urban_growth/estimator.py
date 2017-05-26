from urban_growth.model import *

class estimator(settlement_model):

	def __init__(self, **kwargs):
		
		settlement_model.__init__(self, **kwargs)
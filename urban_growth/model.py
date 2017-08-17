################################################################################
# INITIALIZATION
################################################################################

import numpy as np
from skimage import morphology
from scipy.spatial import distance

from scipy.ndimage.filters import gaussian_filter
from itertools import product
from urban_growth.utility import *
from scipy.ndimage.morphology import distance_transform_edt
from copy import deepcopy
from scipy.special import expit
from scipy.ndimage import convolve
from scipy import special

class settlement_model:
	
################################################################################
# INITIALIZATION
################################################################################

	def __init__(self, M0 = None, geo = None,  N_pix = 100, sigma = 1, t = .5, class_type = 'blur', thresh = 10):
		if M0 is not None:
			self.set_M0(M0 = M0)
		if geo is not None:
			self.geo = geo
		else:
			self.geo = np.ones(self.M0.shape[0], self.M0.shape[1])
		
		self.N_pix = N_pix
		self.sigma = sigma
		self.t     = t
		self.class_type = class_type
		self.thresh = thresh
		self.partition_clusters()
		
	def set_M0(self, M0 = None, **kwargs):
		if M0 is not None:
			self.M0 = M0
		else:
			self.M0 = random_mat(**kwargs)

		self.M = self.M0.copy()
		
	def get_M0(self):
		return self.M0

	def partition_clusters(self):
		if self.class_type == 'blur':
			self.C = gaussian_blur_partition(self.M, self.sigma, self.t)
		# the below is pseudocode
		elif self.class_type == 'thresh':
			self.C = threshold_partition(self.M, self.thresh)
			
	def settlement_rate(self, K, pars, use_grad = False):
		'''
		Wasteful to compute the gradient information when we do not need it. 
		Possible improvement needed for when we get to simulating. Should 
		only be a 2x speedup though, so may not be worth the time. 
		'''
		def f(i):
		    alpha = pars['alpha'][i]
		    gamma = pars['gamma'][i]
		    beta  = pars['beta']

		    k   = K ** (-gamma)
		    d_k = k * np.log(K)

		    denom   = k.sum()
		    d_denom = d_k.sum()

		    convd   = convolve(self.C[i], k, cval = self.C[i].mean())
		    d_convd = convolve(self.C[i], d_k, cval = self.C[i].mean()) 

		    c_deriv = - alpha * (d_convd * denom - d_denom * convd) / (denom ** 2)
		    a_deriv = convd / denom

		    return {'denom' : denom, 'convd' : convd, 'a_deriv' : a_deriv, 'c_deriv' : c_deriv}

		components = [f(i) for i in range(2)]

		rate = expit(pars['beta'] + sum(
		                        pars['alpha'][i] * components[i]['convd'] / components[i]['denom'] 
		                        for i in range(2)
		                        ))
		
		rate = rate * self.geo # compares well to self.settlement_rate(). 
		
		a_derivs = np.array([components[i]['a_deriv'] for i in range(2)])
		c_derivs = np.array([components[i]['c_deriv'] for i in range(2)])
		b_deriv  = np.expand_dims(np.ones(self.M0.shape), 0)
		grad     = np.concatenate((a_derivs, c_derivs, b_deriv)) * rate * (1 - rate)

		if use_grad: 
			return rate, grad
		else:
			return rate

import numpy as np
from skimage import morphology
from scipy.spatial import distance

from scipy.ndimage.filters import gaussian_filter
from itertools import product
from urban_growth.utility import *
from scipy.ndimage.morphology import distance_transform_edt
from copy import deepcopy

class settlement_model:
	
################################################################################
# INITIALIZATION
################################################################################

	def __init__(self, M0 = None, geo = None, model = None):
		if M0 is not None:
			self.set_M0(M0 = M0)
		if geo is not None:
			self.geo = geo
		else:
			self.geo = np.ones(self.M0.shape[0], self.M0.shape[1])
		if model is not None:
			self.set_model(model)
		
	def set_M0(self, M0 = None, **kwargs):
		if M0 is not None:
			self.M0 = M0
		else:
			self.M0 = random_mat(**kwargs)

		self.M = self.M0.copy()
		self.update_morphology()

	def get_M0(self):
		return self.M0

	def update_morphology(self):
		# generate clusters
		mask = morphology.label(self.M, connectivity = 1)
		self.clusters = {i : np.array(np.where(mask == i)).T for i in np.unique(mask) if i > 0}

		# unsettled cells
		a = np.where(self.M == 0)
		self.unsettled = np.array([[a[0][i], a[1][i]] for i in range(len(a[0]))])

		# get areas
		self.areas = {i : self.clusters[i].shape[0] for i in self.clusters}
	
	def set_model(self, model):
		self.model = model
		
################################################################################
# PARTITIONING OF CLUSTER TYPES
################################################################################
	
	def get_types(self):
		return self.types

	def partition_clusters(self, T_vec):
		'''
		Stores the cluster types as a matrix of one-hot vectors. 
		'''
		areas = deepcopy(self.areas)

		d_list = {len(T_vec) : areas}

		i = 0
		for t in T_vec:
		    d_list[i] = {}
		    for k in areas.keys():
		        if areas[k] < t:
		            d_list[i].update({k : areas[k]})
		            areas.pop(k)
		    i += 1

		arr = np.zeros((len(d_list), len(self.areas)))

		for d in d_list:
		    for k in d_list[d]:
		        arr[d, k - 1] = 1

		self.types = arr

################################################################################
# DISTANCE CALCULATIONS
################################################################################

	def make_dist_array(self, trunc):
		'''
		Figure out how to do this by component using the bwdist to enforce the truncation. I expect this would save quite a lot of time and possibly memory. 
		'''

		D = np.zeros((self.types.shape[0], trunc, self.M0.shape[0],self.M0.shape[1] ))

		type_indices = {i : np.concatenate(
		    [self.clusters[j] for j in self.clusters if self.types[i][j-1] == 1]) 
		                for i in range(self.types.shape[0])}

		for k in range(D.shape[0]):
		    ix = type_indices[k]
		    dists = distance.cdist(self.unsettled, ix, 'euclidean').astype(int)
		    for i in range(dists.shape[0]):
		        a = np.unique(dists[i], return_counts=True)
		        d = a[0][a[0] < trunc]
		        f = a[1][a[0] < trunc]
		        D[k, d, self.unsettled[i][0], self.unsettled[i][1]] = f
		        
		self.D = D

	def distance_feature(self, gamma, component = None, deriv = False):
		t = self.D.shape[1]
		t = np.arange(1, t+1)
		
		if component is None:
			t = t[np.newaxis,:]
		
		v = t ** (- 1.0 * np.array([gamma]).T) 
		
		v = np.expand_dims(v, axis = 2)
		v = np.expand_dims(v, axis = 3)

		t = np.expand_dims(t, axis = 2)
		t = np.expand_dims(t, axis = 3)

		if component is not None:
			intermediate = self.D[component] * v
			out = intermediate.sum(axis = 0)
			if deriv:
				df = - (intermediate * np.log(t)).sum(axis = 0)
		else:
			out = (self.D * v).sum(axis = 1)
			if deriv:
				df = - (self.D * v * np.log(t)).sum(axis = 1)
		if deriv:
			return out, df
		
		return out


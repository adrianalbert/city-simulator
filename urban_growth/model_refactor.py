import numpy as np
from skimage import morphology
from scipy.spatial import distance

from scipy.ndimage.filters import gaussian_filter
from itertools import product
from urban_growth.helpers import *
from scipy.ndimage.morphology import distance_transform_edt
from copy import deepcopy

class settlement_model:
	def __init__(self, M0 = None, geo = None):
		if geo is not None:
			self.geo = geo
		if M0 is not None:
			self.set_M0(M0 = M0)

	def update_morphology(self, dist_approx = False, trunc = 50):
		# generate clusters
		mask = morphology.label(self.M, connectivity = 1)
		self.clusters = {i : np.array(np.where(mask == i)).T for i in np.unique(mask) if i > 0}

		# unsettled cells
		a = np.where(self.M == 0)
		self.unsettled = np.array([[a[0][i], a[1][i]] for i in range(len(a[0]))])

		self.areas = {i : self.clusters[i].shape[0] for i in self.clusters}
		if dist_approx:
		# areas	
			self.edt()
		
	def set_M0(self, M0 = None, **kwargs):
		if M0 is not None:
			self.M0 = M0
		else:
			self.M0 = random_mat(**kwargs)

		self.M = self.M0.copy()
		self.update_morphology()

	def get_M0(self):
		return self.M0


	def distance_variables(self):
		# uses an approximation based on the EDT that is generally accurate to within 10% or so. 

		A = np.array([self.areas[i+1] for i in range(len(self.areas))])
		A = np.array([[A]]).T

		a =   np.ones(self.r_0.shape)
		b = - self.r_0 
		c = - self.r_0 ** 2
		d =   self.r_0 ** 3 - A * self.r_0
		
		self.r_1 = np.real(cubic(a, b, c, d))
		self.theta = (self.r_1 - self.r_0) / self.r_0

	def distance_feature(self, gamma, partitioned = False):
		M = distance_approximation(self.r_0, self.r_1, self.theta, gamma)
		if partitioned:
			return np.dot(self.types, np.swapaxes(M, 1, 0))
		else:
			return M

	def get_edt(self, j):

	    M = np.zeros(self.M.shape)

	    for i in range(self.clusters[j].shape[0]):
	        M[tuple(self.clusters[j][i,])] = 1 

	    return distance_transform_edt(1 - M)    

	def edt(self):
		
	    self.r_0 = np.array([self.get_edt(j + 1) for j in range(len(self.clusters))])

	def logistic_components(self, alpha, beta, gamma):
		f = self.distance_feature(np.array([[np.dot(gamma, self.types)]]).T, partitioned = True)
		M = expit(np.array([[alpha]]).T * expit(f) + np.array([[beta]]).T)
		return M

	def logistic_density(self, **kwargs):
		M = self.logistic_components(**kwargs)
		return (M**2).sum(axis = 0) / M.sum(axis = 0)

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

	# full distance calculations

	def make_dist_array(self, trunc):
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

	def dist_array_feature(self, gamma, component = None, deriv = False):
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


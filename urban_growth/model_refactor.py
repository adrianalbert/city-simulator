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

	def update_morphology(self):
		# generate clusters
		mask = morphology.label(self.M, connectivity = 1)
		self.clusters = {i : np.array(np.where(mask == i)).T for i in np.unique(mask) if i > 0}

		# unsettled cells
		a = np.where(self.M == 0)
		self.unsettled = np.array([[a[0][i], a[1][i]] for i in range(len(a[0]))])

		# areas
		self.areas = {i : self.clusters[i].shape[0] for i in self.clusters}
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
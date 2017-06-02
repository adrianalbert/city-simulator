import numpy as np
from skimage import morphology
from scipy.spatial import distance
from scipy.special import expit
from scipy.ndimage.filters import gaussian_filter
from itertools import product
from urban_growth.helpers import *
from scipy.ndimage.morphology import distance_transform_edt

class settlement_model:
	def __init__(self, M0 = None, geo = None):
		if geo is not None:
			self.geo = geo
		if M0 is not None:
			self.set_M0(M0 = M0)

	def set_M0(self, M0 = None, **kwargs):
		if M0 is not None:
			self.M0 = M0
		else:
			self.M0 = random_mat(**kwargs)

		# generate clusters
		mask = morphology.label(self.M0, connectivity = 1)
		self.clusters = {i : np.array(np.where(mask == i)).T for i in np.unique(mask) if i > 0}

		# unsettled cells
		a = np.where(self.M0 == 0)
		self.unsettled = np.array([[a[0][i], a[1][i]] for i in range(len(a[0]))])

		# areas
		self.areas = {i : self.clusters[i].shape[0] for i in self.clusters}

	def distance_variables(self):
		# uses an approximation based on the EDT that is generally accurate to within 10% or so. 

		A = np.array([self.areas[i+1] for i in range(len(self.areas))])
		A = np.array([[A]]).T

		a = np.ones(self.Q.shape)
		b = - self.Q 
		c = - self.Q ** 2
		d = self.Q ** 3 - A * self.Q
		self.r_1 = np.real(cubic(a, b, c, d))
		self.theta = (self.r_1 - self.Q) / self.Q

	def distance_feature(self, gamma):
		return distance_approximation(self.Q, self.r_1, self.theta, gamma)

	# could be generalized later
	def assign_types(self, T):
		self.types = {i : (self.areas[i] > T) * 1 for i in self.areas}		

	# distance feature
	def get_edt(self, j):

	    M = np.zeros(self.M0.shape)

	    for i in range(self.clusters[j].shape[0]):
	        M[tuple(self.clusters[j][i,])] = 1 

	    return distance_transform_edt(1 - M)    

	def edt(self):
	    self.Q = np.array([self.get_edt(j + 1) for j in range(len(self.clusters))])

	def areal_weight(self):
		print 'not implemented'

	



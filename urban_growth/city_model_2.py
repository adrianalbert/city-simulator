import numpy as np
from skimage import morphology
from scipy.spatial import distance


class settlement_model:

	def __init__(self, M0 = None, geo = None, D = None, L = None):

		self.M0 = M0

		if geo is not None:
			self.geo = geo
		else:
			geo = np.ones(self.M0.shape)

		if D is not None: 
			self.D = D

	def simulate(self, model_type, pars, return_trace = True, trace_frequency = 1, trace_dir = ''):
		print 'not implemented'

	def infer(self, model_type, M1, return_trace = True):
		print 'not implemented'

def id_clusters(M, **kwargs):

	mask = morphology.label(M, **kwargs)
	return mask

def cluster_areas(mask):
	
	u = np.unique(mask, return_counts = True)
	area_dict = {u[0][i] : u[1][i] for i in range(len(u[0]))} # needs to be sorted?
	area_dict.pop(0) # ignore background
	return area_dict

def settlement_types(M,  return_type = 'cell', thresh = 0, **kwargs):
	'''
		Compute the types of cells and return the results at either the cluster or cell level. 
	'''

	mask      = id_clusters(M, **kwargs)
	area_dict = cluster_areas(mask)
	
	urban_clusters  = {key : area_dict[key] for key in area_dict if area_dict[key] >= thresh}

	rural_clusters  = area_dict
	for key in urban_clusters:
		rural_clusters.pop(key, None) 

	if return_type == 'cluster':
		return rural_clusters, urban_clusters
	else:
		ix = np.in1d(mask.ravel(), urban_clusters.keys()).reshape(mask.shape)
		a = np.where(ix)

		urban_cells = np.array([[a[0][i], a[1][i]] for i in range(len(a[0]))])

		ix = np.in1d(mask.ravel(), rural_clusters.keys()).reshape(mask.shape)
		a = np.where(ix)

		rural_cells = np.array([[a[0][i], a[1][i]] for i in range(len(a[0]))])

		a = np.where(mask == 0)
		unsettled_cells = np.array([[a[0][i], a[1][i]] for i in range(len(a[0]))])

		return rural_cells, urban_cells, unsettled_cells

def get_dists(M, thresh = 0, **kwargs):
	
	def f(j, arr):
		a = np.unique(arr[j,], return_counts=True)
		return [(a[0][i], a[1][i]) for i in range(a[0].shape[0])]

	rural, urban, unsettled = settlement_types(M, 'cell', thresh, **kwargs)
	
	rural_dist_mat = distance.cdist(unsettled, rural, 'euclidean').astype(int)
	rural_dists    = {i : f(i, rural_dist_mat) for i in range(rural_dist_mat.shape[0])}

	urban_dist_mat = distance.cdist(unsettled, urban, 'euclidean').astype(int)
	urban_dists    = {i : f(i, urban_dist_mat) for i in range(urban_dist_mat.shape[0])}

	return rural_dists, urban_dists

def distance_weights(M, dists, gamma_r, gamma_u, thresh, **kwargs):
	def f(i, gamma_r, gamma_u):
		
		a = np.array(dists[0][i])
		r = np.dot(a[:,0] ** (-gamma_r), a[:,1])

		b = np.array(dists[1][i])
		u = np.dot(b[:,0] ** (-gamma_u), b[:,1])
		
		return r, u
	
	rural, urban, unsettled = settlement_types(M, 'cell', thresh, **kwargs)
	
	weights_rural = np.zeros(M.shape)
	weights_urban = np.zeros(M.shape)

	weight_tuples = [f(i, gamma_r, gamma_u) for i in range(len(dists[0]))]

	for i in range(unsettled.shape[0]):
		weights_urban[unsettled[i][0], unsettled[i][1]] = weight_tuples[i][1]
		weights_rural[unsettled[i][0], unsettled[i][1]] = weight_tuples[i][0]

	return weights_rural, weights_urban






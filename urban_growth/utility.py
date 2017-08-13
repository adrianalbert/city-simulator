import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage import morphology
from itertools import product
from scipy.spatial import distance

def random_mat(L, density = .5, blurred = True, blur = 3, central_city = True):
	
	M = np.random.rand(L, L)

	if central_city:

		M[(1 * L/2 - L / 10):(1 * L/2 + L/10),(1 * L/2 - L / 10):(1 * L/2 + L/10)] = 0

	if blurred: 
		M = gaussian_filter(M, blur)
		
	ix_low = M < density  # Where values are low
	M[ix_low]  = 1

	M[M < 1] = 0
		
	return M

def normalizer(gamma, extent = 50, deriv = False):
	# assumes centered -- it may be useful to consider near the corners down the line. 
	# this difference will mostly matter for low gamma. 
	
	# consider whether we can do this in one calculation instead of two down the line 
	# when we start thinking about performance again. 
	
	if extent is None:
		extent = 50

	x = np.arange(0, extent + 1) ** 2.0
	y = np.arange(1, extent + 1) ** 2.0
	xy = product(x, y)
	norm = 4 * sum([np.sqrt(x + y) ** (-gamma)  for (x, y) in xy])
	if deriv:
		x = np.arange(0, extent + 1) ** 2.0
		y = np.arange(1, extent + 1) ** 2.0
		xy = product(x, y)
		dnorm = - 4 * sum([np.sqrt(x + y) ** (-gamma) * np.log(np.sqrt(x + y)) for (x, y) in xy])
		return norm, dnorm
	return norm

def distance_kernel(L, base_val = .5, unit = 1):
	
	M = np.zeros((2*L + 1, 2*L + 1))
	ix_M = np.where(M > -1)
	ix_arr_M = np.array([ix_M[0], ix_M[1]]).T / unit
	dists = distance.cdist(ix_arr_M, np.array([[L, L]]))
	M = dists.reshape(M.shape)
	M[L, L] = base_val
	return M
	
def gaussian_blur_partition(M, sigma, t):
	m    = gaussian_filter(M, sigma)
	C    = np.zeros((2, m.shape[0], m.shape[1]))
	C[0] = (m < t)  * M  # rural
	C[1] = (m >= t) * M # urban
	return C

def threshold_partition(M, thresh):
	
	morph  = morphology.label(M > 0)
	C      = np.zeros((2, morph.shape[0], morph.shape[1]))
	labels = np.unique(morph)
	sizes  = {lab : M[np.where(morph == lab)].sum() for lab in labels}

	for lab in sizes:
		if sizes[lab] >= thresh:
			C[1][np.where(morph == lab)] = 1
		else:
			C[0][np.where(morph == lab)] = 1
	
	return C
			
			



def to_vec(pars):
	return np.concatenate((pars['alpha'], pars['gamma'], np.array([pars['beta']])))

def from_vec(pars_vec):
	d = {'alpha' : pars_vec[0:2],
		 'gamma' : pars_vec[2:4],
		 'beta'  : pars_vec[4]}
	return d
	

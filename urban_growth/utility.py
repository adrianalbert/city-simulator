import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage import morphology
from itertools import product

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
	

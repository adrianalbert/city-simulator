import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage import morphology
from scipy.special import expit
from itertools import product

# DENSITIES AND GRADIENTS



## Logistic model specification
def logistic_components(m, alpha, beta, gamma):
	f = m.distance_feature(np.array([[np.dot(gamma, m.get_types())]]).T, 
						   partitioned = True) 
	M = expit(np.array([[alpha]]).T * expit(f) + np.array([[beta]]).T)
	return M

def logistic_density(m, **kwargs):
	M = logistic_components(m, **kwargs)
	return (M**2).sum(axis = 0) / M.sum(axis = 0)

def logistic_gradient_term(m, p = None, use_beta = True, **pars):

	# typically should pass components since they have already 
	# been calculated, but in case not....
	if p == None:
		p = logistic_components(m, **pars)

	# construct distance features
	gamma = pars['gamma']
	f_gamma   = m.distance_feature(np.array([[np.dot(gamma, m.get_types())]]).T, 
								   partitioned = True) 
	f_gamma_1 = m.distance_feature(np.array([[np.dot(gamma + 1, m.get_types())]]).T, 
								   partitioned = True) 

	# extract parameters and convert to nicely shaped arrays
	alpha_arr = np.array([[pars['alpha']]]).T
	
	
	beta_arr  = np.array([[pars['beta' ]]]).T
	gamma_arr = np.array([[pars['gamma']]]).T

	# compute gradient components

	d_alpha = p * (1 - p) * f_gamma
	if use_beta:
		d_beta =  p * (1 - p)
	else:
		d_beta = np.zeros(shape = p.shape)

	d_gamma   = - alpha_arr * gamma_arr * p * (1 - p) * f_gamma_1

	# arrange as nice array and return
	dp = np.array([d_alpha, d_beta, d_gamma])

	return dp

def restricted_logistic_gradient_term(m, p, **pars):
	return logistic_gradient_term(m, p, use_beta = False, **pars) 

## Linear model specification
def linear_components(m, alpha, gamma):
	f = m.distance_feature(np.array([[np.dot(gamma, m.get_types())]]).T, 
						   partitioned = True) 
	return np.array([[alpha]]).T * f / np.array([[normalizer(gamma, extent = None)]]).T
	
def linear_density(m, **kwargs):
	M = linear_components(m, **kwargs)
	return (M**2).sum(axis = 0) / M.sum(axis = 0)

def linear_gradient_term(m, p = None, **kwargs):
	if p == None:
		p = linear_components(m, **kwargs)

	gamma = kwargs['gamma']
	f_gamma_1 = m.distance_feature(np.array([[np.dot(gamma + 1, m.get_types())]]).T, 
								   partitioned = True) 

	alpha_arr = np.array([[kwargs['alpha']]]).T
	gamma_arr = np.array([[kwargs['gamma']]]).T

	d_alpha = p / alpha_arr
	d_gamma = alpha_arr * gamma_arr * (np.array([[normalizer(gamma + 1)]]).T - f_gamma_1) / np.array([[normalizer(gamma) ** 2]]).T

	dp = np.array([d_alpha, d_gamma])
	return dp

## Model types
models = {
		  'logistic': {'components'      : logistic_components, # sigmoid model
					   'density'         : logistic_density, 
					   'gradient_term'   : logistic_gradient_term},

		  'linear'  : {'components'      : linear_components, # ema's original model
					   'density'         : linear_density, 
					   'gradient_term'   : linear_gradient_term},

		   'restricted_logistic'  : {'components'    : logistic_components, 
					   				 'density'       : logistic_density, 
					   				 'gradient_term' : restricted_logistic_gradient_term},
		 }

## gradient computation
def gradient(m, X, model = 'logistic', **pars):
	
	p = models[model]['components'](m, **pars)

	coef = X / p - (1 - X) / (1 - p)
	
	dp = models[model]['gradient_term'](m, p, **pars)

	dq = dp / np.array(p) - np.nansum(dp, axis = 1, keepdims=True) / np.nansum(p, axis = 0, keepdims=True)

	grad = dq + coef * dp

	return grad

def to_array(**kwargs):
	return np.array([kwargs[k] for k in kwargs])

def from_array(arr):
	'''
	Not very flexible at this stage
	'''
	d = {'alpha' : arr[0], 'gamma' : arr[-1]}
	if arr.shape[0] > 2:
		d.update({'beta' : arr[1]})
	return d

## UTILS
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

def cubic(a, b, c, d):
	d0 = b ** 2 - 3 * a * c
	d1 = 2*b**3 - 9 * a * b * c + 27 * a**2 * d 
	
	zeta = -0.5 + (0.5j * np.sqrt(3))
	C = ((d1 + np.sqrt(d1**2 - 4*d0**3 + 0j))/2.0 + 0j)**(1.0/3.0)
	C1 = C * zeta

	x = - 1 / (3.0 * a) * (b + C1 + d0 / C1)
	return x

def distance_approximation(r_0, r_1, theta, gamma):
	
	return 2 * theta * (r_0 ** (2.0 - gamma) - r_1 ** (2.0 - gamma)) / (gamma - 2.0)

def normalizer(gamma, extent = 50):
	# assumes centered -- it may be useful to consider near the corners down the line. 
	# this difference will mostly matter for low gamma. 
	
	if extent is None:
		extent = 50

	x = np.arange(0, extent + 1) ** 2.0
	y = np.arange(1, extent + 1) ** 2.0
	xy = product(x, y)
	return 4 * sum([np.sqrt(x + y) ** (-gamma)  for (x, y) in xy])

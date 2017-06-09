import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage import morphology
from scipy.special import expit
from itertools import product

# DENSITIES AND GRADIENTS


def distance_feature_calc(m, gamma, dist_approx = False, component = None):
	if component is not None:
		types = m.get_types()[component]
	else:
		types = m.get_types()
	if dist_approx:
		f = m.distance_feature(np.array([[np.dot(gamma, types)]]).T, 
							   partitioned = True) 
	else:
		f = m.dist_array_feature(gamma, component)

	return f

## Logistic model specification
def logistic_components(m, alpha, beta, gamma, dist_approx = False):
	
	f = distance_feature_calc(m, gamma, dist_approx)

	M = expit(np.array([[alpha]]).T * expit(f) + np.array([[beta]]).T)
	return M

def logistic_density(m, **kwargs):

	f = distance_feature_calc(m, gamma, dist_approx)
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
def linear_components(m, alpha, gamma, dist_approx = False, use_grad = False, component = None):
	
	gamma = 1.0 * gamma
	# f = distance_feature_calc(m, gamma, dist_approx, component)
	if use_grad:
		f, df = m.dist_array_feature(gamma, component, deriv = True)
		norm, dnorm = normalizer(gamma, deriv = True)
	else:
		f = m.dist_array_feature(gamma, component, deriv = False)
		norm = normalizer(gamma)
	
	# component itself
	c = np.array([[alpha]]).T * f / np.array([[norm]]).T
	
	if component is None:
		norm = np.expand_dims(norm, axis = 1)
		norm = np.expand_dims(norm, axis = 2)
		if use_grad:
			dnorm = np.expand_dims(dnorm, axis = 1)
			dnorm = np.expand_dims(dnorm, axis = 2)

	if use_grad:
		d_alpha = c / np.expand_dims(np.expand_dims(alpha, 1), 1)
		d_gamma = (df * norm - dnorm * f) / norm ** 2	
		grads = np.concatenate((d_alpha[np.newaxis,], d_gamma[np.newaxis,]))
		return c, grads
	
	else:
		return c
	
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

def linear_mixture_density(m, **pars):
	pi = pars['pi']
	pars.pop('pi')

	M = linear_components(m, **pars)
	return (np.array([[pi]]).T * M).sum(axis = 0)  # check broadcasting

def linear_mixture_gradient_term():
	print 'not implemented'

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
		   
		   'linear_mixture' :       {'components'    : linear_components, 
					   				 'density'       : linear_mixture_density, 
					   				 'gradient_term' : linear_mixture_gradient_term},
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

def normalizer(gamma, extent = 50, deriv = False):
	# assumes centered -- it may be useful to consider near the corners down the line. 
	# this difference will mostly matter for low gamma. 
	
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
	

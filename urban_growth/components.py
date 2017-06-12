import numpy as np
from scipy.special import expit
from utility import *

################################################################################
# MODEL SPECIFICATIONS 
################################################################################

def logistic_density(m, alpha, gamma, beta, use_grad = False):
	'''
	Alpha and gamma are vectors, beta is a scalar
	get gradient later
	'''
	if use_grad: 
		c, d_theta = linear_components(m, alpha, gamma, use_grad = True)
	else:
		c = linear_components(m, alpha, gamma)
	
	M = expit(np.nansum(c, axis = 0) + beta)

	if use_grad:	

		# d_theta = np.transpose(d_theta, (1, 0, 2, 3))
		n_theta = d_theta.shape[0] * d_theta.shape[1]
		d_theta = np.reshape(d_theta, (n_theta, m.M0.shape[0], m.M0.shape[1]))
		d_beta = np.ones((m.M0.shape[0], m.M0.shape[1]))
		

		d_beta = np.expand_dims(d_beta, axis = 0)

		d_pars = np.concatenate((d_theta, d_beta)) * M * (1 - M)
		return M, d_pars

	return M

## Linear model specification, think this one is ok
def linear_components(m, alpha, gamma, use_grad = False, component = None):
	
	gamma = 1.0 * gamma
	# f = distance_feature(m, gamma, component)
	if use_grad:
		f, df = m.distance_feature(gamma, component, deriv = True)
		norm, dnorm = normalizer(gamma, extent = m.trunc, deriv = True)
	else:
		f = m.distance_feature(gamma, component, deriv = False)
		norm = normalizer(gamma, extent = m.trunc)
	
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
		d_gamma = np.expand_dims(np.expand_dims(alpha, 1), 2) * (df * norm - dnorm * f) / norm ** 2	
		grads = np.concatenate((d_alpha[np.newaxis,], d_gamma[np.newaxis,]))
		return c, grads
	
	else:
		return c

# deprecate both density functions in favor of:
def density(m, model, **pars):
	if model == 'linear':
		pi = pars['pi']
		pars.pop('pi')

		M = models[model]['components'](m, **pars)
		return (np.array([[pi]]).T * M).sum(axis = 0)  	
	elif model == 'logistic':
		return logistic_density(m, **pars)



################################################################################
# HELPER FUNCTIONS FOR MODELS
################################################################################

models = {
		  
		  'linear'  : {'components' : linear_components, # ema's original model
					   'par_names'  :  ['alpha', 'gamma', 'pi']}
		  }

def to_array(pars, model):
	if model not in models:
		print 'model not recognized'
		return None 
	return np.array([pars[k] for k in models[model]['par_names']])

def from_array(arr, model):
	return {models[model]['par_names'][k] : arr[k] for k in range(arr.shape[0])}

def to_vec(arr, model = None):
	return arr.reshape(-1)

def from_vec(v, model):
	ell = len(v)
	m = len(models[model]['par_names'])
	n = ell / m
	return v.reshape(m, n)
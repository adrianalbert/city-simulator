import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage import morphology

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



# -------------------------------------------------
# HOW TO RUN THIS SCRIPT
# -------------------------------------------------

# This is a first draft of a simulator script for Ema to use as we study the 
# physical properties of samples from this model. It should run on a CLEAN 
# and UP TO DATE installation of Python 2. The package requirements are 
# documented in requirements.txt

# To run this script: 
# 	1. Update simulator_files/pars.csv with the set of parameters you want to run. 
#      If there are a lot, it may be best to do this procedurally, e.g. in Matlab.
# 	2. Generate M0.txt (initial condition) and geo.txt (geographic mask) as matrices 
#      saved in plaintext. M0 should have 1s where cells are settled and 0 otherwise. 
# 	   geo should have 0s where cells cannot be settled. 1 indicates no impediment
#      to settlement; values between 0 and 1 are allowed. These two files should 
#      sit in simulator_files/
#   3. Adjust the parameters below as appropriate. 
#   4. Open a terminal, and type:
#      > python simulation_script.py  
#   5. The terminal will print out progress messages after it has completed each simulation. 

# The output of this script is a set of matrices saved in plaintext in the 
# simulator_files/output/ folder. A 0 indicates no settlement. A number larger
# than 0 indicates that the pixel was settled in the corresponding round. 
# E.g. a 1 means that the pixel was settled in M0, a 2 means that the pixel 
# was settled in the first simulation round, etc. 
# If you choose viz = True below, then the script will also construct a set of 
# simple visualizations of these matrices in the simulator_files/viz/ folder. 

# Each file has a label, e.g. M_1.txt and (optionally) M_1.png. The numeric 
# suffix refers to the row of pars.csv with the parameters that generated that 
# matrix. E.g. M_70.txt and M_70.png were generated by the parameters in the 70th
# row (excluding headers) of pars.csv

# -------------------------------------------------
# PARAMETERS TO ADJUST
# -------------------------------------------------

viz           = True # if True, will make simple visualizations in viz/
n_iters       = 5    # number of time-steps
distance_unit = 2.5  # number of pixels per kilometer

# -------------------------------------------------
# MAIN SCRIPT, DON'T TOUCH
# -------------------------------------------------

import numpy as np
from urban_growth.simulator import *
from urban_growth.components import *
from matplotlib import pyplot as plt
import pandas as pd
import os

# load data
geo = np.loadtxt('simulator_files/geo.txt')
M0  = np.loadtxt('simulator_files/M0.txt')

# load parameter list to simulate
pars_df = pd.read_csv('simulator_files/pars.csv')

# check for out folder presences and create if not there
out_dirs = ['simulator_files/output']
if viz:
    out_dirs.append('simulator_files/viz')

for d in out_dirs:
    if not os.path.isdir(d):
       os.makedirs(d)

# main loop
for i in range(len(pars_df)):
    
    print 'running parameter set ' + str(i + 1)
    
    # construct parameter dict
    pars_vec = pars_df.iloc[i]
    pars = {'alpha' : np.array([pars_vec[0], pars_vec[1]]),
            'gamma' : np.array([pars_vec[2], pars_vec[3]]), 
            'beta'  : np.array([pars_vec[4]])}
    T = [pars_vec[5]]
    
    # construct simulator
    s = simulator(M0 = M0, 
              geo = geo, 
              model = 'logistic', 
              unit = distance_unit)
    
    s.update_morphology()
    s.make_dist_array()
    s.partition_clusters(T)
    s.partition_dist_array()
    
    # run the sim
    M = s.dynamics(T, n_iters = n_iters, verbose = False, **pars)
    
    np.savetxt('simulator_files/output/M_' + str(i + 1) + '.txt',M.astype(int), fmt = '%i')
    
    if viz: 
        string_alpha = r'$\alpha_r = $' + str(pars['alpha'][0]) + r', $\alpha_u = $' + str(pars['alpha'][1]) + ', '
        string_gamma = r'$\gamma_r = $' + str(pars['gamma'][0]) + r', $\gamma_u = $' + str(pars['gamma'][1]) + ', '
        string_beta  = r'$\beta = $' + str(pars['beta'][0]) + ', '
        string_T     = r'$T = $' + str(T[0])
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(M)
        plt.title(string_alpha + string_gamma + string_beta + string_T)

#         fig.text(.2,.0,string_alpha + string_gamma + string_beta + string_T)
        plt.colorbar(im)
        plt.savefig('simulator_files/viz/M_' + str(i+1) + '.png')
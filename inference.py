import numpy as np
from urban_growth.model import *
from urban_growth.simulator import *
from urban_growth.estimator import *
import os 
import pandas as pd
import pickle

# main parameters
Q = [0, .1, .25, .4]

L = 20

unit = 3.333 # pixels per km
base_val = .5

K = distance_kernel(L, base_val, unit)

par_0 = {'alpha' : np.array([100, 30]),
        'gamma' : np.array([1.2, 2.5]),
        'beta'  : -25}


# construct control frame
files = os.listdir('data')
maps = [name for name in files if 'mask' not in name]
control_df = pd.DataFrame({'file_name' : maps} )

control_df['city'] = control_df.file_name.str[0:3]
control_df
control_df['year_begin'] = pd.to_numeric(control_df.file_name.str[3:7])
control_df = control_df[control_df.year_begin < 2011]

def get_year_end(year):
    if year in [1990, 2000]:
        return year + 10
    else:
        return year + 5
    
v_gye = np.vectorize(get_year_end)

control_df['year_end'] = v_gye(control_df.year_begin)

control_df = control_df.drop('file_name', 1)

# Define grid search

def grid_search(city, year_begin, year_end, Q, par_0):

        # get and clean data for given city and years
        M0  = np.loadtxt('data/' + city + str(year_begin) + '.csv', dtype=float, delimiter=',')
        M1  = np.loadtxt('data/' + city + str(year_end) + '.csv',   dtype=float, delimiter=',')
        geo = np.loadtxt('data/' + city + '_mask.csv',   dtype=float, delimiter=',')

        geo = 1 - geo
        M0 = M0 * geo
        M1 = M1 * geo

        M1  = np.maximum(M0, M1)

        # initialize lists

        city_vec       = []
        year_begin_vec = []
        year_end_vec   = []
        q_vec          = []
        T_vec          = []
        par_0_vec      = []
        par_vec        = []
        ll_vec         = []
        cov_vec        = []
        N_eff_vec      = []
        n_pars_vec     = []
        AIC_vec        = []
        rate_vec       = [] 
        bg_rate_vec    = []

        for q in Q:
            M = M0.copy()
            N = M1.copy()
            M[M < q] = 0
            N[N < q] = 0

            morph = morphology.label(M > 0)
            C     = np.zeros((2, morph.shape[0], morph.shape[1]))
            labels = np.unique(morph)
            size_thresh = {lab : M[np.where(morph == lab)].sum() for lab in labels}

            size_thresh = np.unique(np.round(size_thresh.values()))

            for T in size_thresh:
                print city + ', ' + str(year_begin) + '-' + str(year_end) + ' : T = ' + str(T) + ' , q = ' + str(q)

                e = estimator(M0 = M, geo = geo, T = T)
                res = e.ml(N, K, par_0, opts = {'disp' : False}, use_grad = True)

                if q == 0:
                    n_pars = 4
                else:
                    n_pars = 7

                ll = e.log_likelihood(K = K, M1 = M1, pars = res[0])

                N_eff = ((1 - M) * geo).sum()
                settlement_rate = (N - M).sum() / N_eff
                bg_rate = expit(res[0]['beta'])

                # updates to storage lists
                city_vec.append(city)
                year_begin_vec.append(year_begin)
                year_end_vec.append(year_end)
                q_vec.append(q)
                T_vec.append(T)
                par_0_vec.append(par_0)
                par_vec.append(res[0])
                ll_vec.append(ll)
                cov_vec.append(res[2])
                N_eff_vec.append(N_eff)
                n_pars_vec.append(n_pars)
                AIC_vec.append(2 * (n_pars - ll))
                rate_vec.append(settlement_rate)
                bg_rate_vec.append(bg_rate)

        df = pd.DataFrame({
            'city'       : city_vec,
            'year_begin' : year_begin_vec,
            'year_end'   : year_end_vec,
            'q'          : q_vec,
            'T'          : T_vec,
            'par_0'      : par_0_vec,
            'par'        : par_vec,
            'll'         : ll_vec,
            'cov'        : cov_vec,
            'N_eff'      : N_eff_vec,
            'n_pars'     : n_pars_vec,
            'AIC'        : AIC_vec,
            'settlement_rate' : rate_vec,
            'bg_rate' : bg_rate_vec
        })

        return df

out_thresh = pd.concat([grid_search(control_df.city.iloc[i], 
                                        control_df.year_begin.iloc[i],
                                        control_df.year_end.iloc[i], 
                                        Q, 
                                        par_0) for i in range(len(control_df))])
out_path = "throughput/out_thresh.p"
out_file = open(out_path,'wb')
pickle.dump(out_thresh,out_file)   
out_file.close()
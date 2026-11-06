import numpy as np 
from read_supernova_data import load_supernova_data
from cosmology_model import mu_array

data = np.asarray(load_supernova_data("data/Supernovadaten.txt"), dtype = float)

z_data = data[:, 0]
mu_obs = data[:, 1]
sigma_mu = data[:, 2]

def negloglike(params):
    Om, Ol, Ok = params
    Or = 1 - Om - Ol - Ok
    
    # Negativer Strahlungsanteil ist physikalisch nicht sinnvoll
    if Or < 0:
        return np.inf

    try:
        mu_model = mu_array(
            z_data, 
            Om, 
            Ol, 
            Ok, 
            n_grid=10000
        )
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return np.inf
    
    residuals = (mu_obs - mu_model) / sigma_mu
    
    return 0.5 * np.sum(
        np.log(2 * np.pi * sigma_mu**2) + residuals**2
    )
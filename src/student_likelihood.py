import numpy as np
from scipy.special import gammaln

from read_supernova_data import load_supernova_data
from cosmology_model import mu_array


data = np.asarray(
    load_supernova_data("data/Supernovadaten.txt"),
    dtype=float
)

z_data = data[:, 0]
mu_obs = data[:, 1]
sigma_mu = data[:, 2]


def negloglike_student(params, nu):
    """
    Negative Log-Likelihood mit Student-t-Verteilung.

    params = [Omega_m, Omega_Lambda, Omega_k]
    nu     = Freiheitsgrade
    """

    Om, Ol, Ok = params

    # Normierungsbedingung:
    Or = 1.0 - Om - Ol - Ok

    if Om < 0 or Or < 0:
        return np.inf

    if nu <= 0:
        return np.inf

    try:
        mu_model = mu_array(
            z_data,
            Om,
            Ol,
            Ok,
            n_grid=5000
        )
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return np.inf

    residuals = (
        mu_obs - mu_model
    ) / sigma_mu

    log_norm = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(sigma_mu)
    )

    log_kernel = (
        -0.5 * (nu + 1.0)
        * np.log1p(residuals**2 / nu)
    )

    loglike = np.sum(log_norm + log_kernel)

    return -loglike
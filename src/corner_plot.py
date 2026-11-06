import numpy as np 
import corner
import matplotlib.pyplot as plt 

from likelihood import negloglike
#from student_likelihood import negloglike_student


#nu = 4.0  # Freiheitsgrade der Student-t-Verteilung


samples = np.loadtxt("results/mcmc_samples.txt")
Om = samples[:, 0]
Ol = samples[:, 1]
Ok = samples[:, 2]

Or = 1 - Om - Ol - Ok

samples_with_radiation = np.column_stack((Om, Ol, Ok, Or))
labels = [
    r"$\Omega_m$",
    r"$\Omega_{\lambda}$",
    r"$\Omega_{\kappa}$",
    r"$\Omega_r$"
]

fig = corner.corner(
    samples_with_radiation,
    labels=labels,
    show_titles=True,
    title_fmt=".4f",
    title_kwargs={"fontsize": 12},
    quantiles=[0.16, 0.5, 0.84],
    label_kwargs={"fontsize": 14},
    plot_contours=True,
    plot_density=True,
    fill_contours=True,
    plot_datapoints=False,
    levels=[0.68, 0.95],
    color="navy",
    smooth=1.0,
    smooth1d=1.0,
)

plt.show()
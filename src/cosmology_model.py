
import numpy as np
from scipy.integrate import cumulative_trapezoid

H0 = 70
c = 299792.458


def mu_array(z_values, Om, Ol, Ok, n_grid=10000):

    z_values = np.asarray(z_values, dtype=float)

    if not np.all(np.isfinite(z_values)):
        raise ValueError("Ungültige Rotverschiebungen.")

    if np.any(z_values < 0):
        raise ValueError("Negative Rotverschiebung.")

    Or = 1.0 - Om - Ok - Ol

    if Or < 0:
        raise ValueError("Or < 0")

    z_grid = np.linspace(0.0, np.max(z_values), n_grid)
    zp = 1.0 + z_grid

    e2 = (
        Om * zp**3
        + Or * zp**4
        + Ol
        + Ok * zp**2
    )

    # E² muss überall strikt positiv sein
    if not np.all(np.isfinite(e2)) or np.any(e2 <= 0):
        raise ValueError("E²(z) <= 0 oder nicht endlich.")

    integral_grid = cumulative_trapezoid(
        1.0 / np.sqrt(e2),
        z_grid,
        initial=0
    )

    integral = np.interp(z_values, z_grid, integral_grid)

    if Ok < 0:
        dm = (
            c / (H0 * np.sqrt(-Ok))
            * np.sin(np.sqrt(-Ok) * integral)
        )

    elif Ok > 0:
        dm = (
            c / (H0 * np.sqrt(Ok))
            * np.sinh(np.sqrt(Ok) * integral)
        )

    else:
        dm = c / H0 * integral

    distance = (1.0 + z_values) * dm

    # Luminositätsdistanz muss positiv und endlich sein
    if not np.all(np.isfinite(distance)) or np.any(distance <= 0):
        raise ValueError("Ungültige Luminositätsdistanz.")

    return 5.0 * np.log10(distance) + 25.0
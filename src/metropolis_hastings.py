import numpy as np
import numdifftools as nd
from likelihood import negloglike
from best_fit import find_best_fit
from scipy.optimize import minimize

N_Samples = 100000
Burn_in = 5000
Output_file = "results/mcmc_samples.txt"

def transformed_to_physical_params(params):
    Om, Ol, eta = params
    Or = np.exp(eta)  # Strahlungsanteil als exp(eta) parametrisiert
    Ok = 1 - Om - Ol - Or
    return np.array([Om, Ol, Ok])

def physical_to_transformed_params(params):
    Om, Ol, Ok = params
    Or = 1.0 - Om - Ol - Ok
    if Or <= 0:
        raise ValueError("Ungültige physikalische Parameter: Or muss > 0 sein.")
    eta = np.log(Or)
    return np.array([Om, Ol, eta])

def negloglike_transformed(params):
    Om, Ol, eta = params
    Or = np.exp(eta)
    Ok = 1.0 - Om - Ol - Or
    
    if Om < 0:
        return np.inf
    
    physical_params = np.array([Om, Ol, Ok])
    
    value = negloglike(physical_params)
    
    if not np.isfinite(value):
        return np.inf
    
    return value

# Grenzen setzen für die transformierten Parameter
def valid_params_transformed(params):
    Om, Ol, eta = params

    if not np.all(np.isfinite(params)):
        return False

    if Om < 0:
        return False

    # Verhindert numerischen Überlauf bei exp(eta)
    if eta > 700:
        return False

    return True

# log für transformierte Parameter 
def neglogposterior_transformed(params):

    if not valid_params_transformed(params):
        return np.inf

    Om, Ol, eta = params

    try:
        value = negloglike_transformed(params)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return np.inf

    if not np.isfinite(value):
        return np.inf

    # Jacobian für einen konstanten Prior in den
    # ursprünglichen Parametern (Om, Ol, Ok)
    return value - eta

# best fit in transformierten Parametern berechnen
def find_best_fit_transformed(initial_guess):

    print("Starte Optimierung im transformierten Parameterraum...")
    print()

    initial_transformed = physical_to_transformed_params(
        np.asarray(initial_guess, dtype=float)
    )

    result = minimize(
        negloglike_transformed,
        x0=initial_transformed,
        method="Nelder-Mead",
        options={
            "maxiter": 10000,
            "xatol": 1e-9,
            "fatol": 1e-9
        }
    )

    return result



def calculate_covariance(best_fit_transformed):

    print()
    print("Berechne Hesse-Matrix am Best-fit Punkt...")
    print()

    hessian = nd.Hessian(
        neglogposterior_transformed,
        method="central",
        step=1e-4
    )

    H = hessian(best_fit_transformed)

    H = 0.5 * (H + H.T)

    print("Hesse-Matrix:")
    print(H)
    print()

    eigenvalues = np.linalg.eigvalsh(H)

    print("Eigenwerte der Hesse-Matrix:")
    print(eigenvalues)
    print()

    if np.any(eigenvalues <= 0):
        raise RuntimeError(
            "Hesse-Matrix ist nicht positiv definit."
        )

    covariance = np.linalg.inv(H)
    covariance = 0.5 * (covariance + covariance.T)

    print("Kovarianzmatrix:")
    print(covariance)
    print()

    return covariance

def metropolis_hastings(
    initial_params,
    proposal_covariance,
    n_samples
):
    ndim = len(initial_params)
    
    samples = np.zeros((n_samples, ndim))
    samples[0] = initial_params
    
    current_params = initial_params.copy()
    current_loglike = -neglogposterior_transformed(current_params)
    
    accepted = 0
    rejected = 0
    rejected_invalid = 0
    
    # --------------------------------------------------------
    # Vorschlagskovarianz skalieren
    #
    # Für einen d-dimensionalen Random-Walk-Metropolis-
    # Algorithmus ist ungefähr 2.38^2 / d ein sinnvoller
    # Startpunkt.
    # --------------------------------------------------------
    
    scaling_factor = 2.38**2 / ndim
    proposal_covariance = proposal_covariance * scaling_factor
    
    # Sonderbehebung für eta
    # die vorschlagskovarianz für eta sollte kleiner sein, 
    # um unphysikalische Vorschläge zu vermeiden
    
    eta_step = 0.1
    proposal_covariance[2, 2] = eta_step**2
    proposal_covariance[2, 0] = 0.0
    proposal_covariance[2, 1] = 0.0
    proposal_covariance[0, 2] = 0.0
    proposal_covariance[1, 2] = 0.0
    
    # Kleine Regularisierung gegen numerische Probleme
    proposal_covariance += 1e-10 * np.eye(ndim)
    
    
    print("Vorschlagskovarianzmatrix:")
    print(proposal_covariance)
    print()
    
    for i in range(1, n_samples):
        # Vorschlag generieren
        proposal = np.random.multivariate_normal(
            mean=current_params,
            cov=proposal_covariance
        )
        
        if not valid_params_transformed(proposal):
            samples[i] = current_params
            rejected_invalid += 1
            continue
        
        proposal_loglike = -neglogposterior_transformed(proposal)
        
        if not np.isfinite(proposal_loglike):
            samples[i] = current_params
            rejected_invalid += 1
            continue
        
        # Akzeptanzwahrscheinlichkeit berechnen
        log_acceptance_ratio = proposal_loglike - current_loglike
        
        if np.log(np.random.rand()) < log_acceptance_ratio:
            # Vorschlag akzeptieren
            current_params = proposal
            current_loglike = proposal_loglike
            accepted += 1
        else:
            # Vorschlag ablehnen
            rejected += 1
        
        samples[i] = current_params
        
        # Fortschritt
        if i % 1000 == 0:
            j = i // 1000
            print(f"Iteration {j:6d} / {n_samples // 1000:6d} abgeschlossen.")
    
    
    total = accepted + rejected + rejected_invalid
    
    acceptance_rate = accepted / (n_samples - 1)
    
    print()    
    print(f"Akzeptiert:          {accepted}")
    print(f"Abgelehnt:           {rejected}")
    print(f"Ungültige Vorschläge:{rejected_invalid}")
    print(f"Akzeptanzrate:       {accepted / total:.4%}")
    print(f"Akzeptanzrate: {acceptance_rate:.3f}")
    
    valid_total = accepted + rejected
    
    if valid_total > 0:
        print(f"Akzeptanzrate (gültige Vorschläge): {accepted / valid_total:.4%}")
        
    return samples

if __name__ == "__main__":
    print("=========================================")
    print("Starte MCMC mit Metropolis-Hastings...")
    print("=========================================")
    print()
    
    result = find_best_fit_transformed(initial_guess=[0.29, 0.69, 0.01])
    
    if not result.success:
        raise RuntimeError("Optimierung fehlgeschlagen: " + result.message)
    
    best_fit_transformed = result.x
    
    best_fit_params = transformed_to_physical_params(
        best_fit_transformed
    )
    
    Om, Ol, Ok = best_fit_params
    Or = np.exp(best_fit_transformed[2])
    print("Best-fit Parameter:")
    print("-----------------")
    print(f"Omega_m      = {best_fit_params[0]:.8f}")
    print(f"Omega_Lambda = {best_fit_params[1]:.8f}")
    print(f"Omega_k      = {best_fit_params[2]:.8f}")  
    print(f"Omega_r      = {Or:.8e}")
    
    covariance = calculate_covariance(best_fit_transformed)
        
    samples_transformed = metropolis_hastings(
        initial_params=best_fit_transformed,
        proposal_covariance=covariance,
        n_samples=N_Samples
    )
    
    # --------------------------------------------------------
    # Burn-in entfernen
    # --------------------------------------------------------
    
    samples_transformed = samples_transformed[Burn_in:]
    print()
    print(f"Samples nach Burn-in: {len(samples_transformed)}")
    
    #--------------------------------------------------------
    # Transformiere Samples zurück in physikalische Parameter
    #--------------------------------------------------------
    samples = np.array([
        transformed_to_physical_params(s) 
        for s in samples_transformed
    ])
    
    # --------------------------------------------------------
    # Samples speichern
    # --------------------------------------------------------
    
    np.savetxt(Output_file, samples)
    
    print()
    print(f"Samples wurden in '{Output_file}' gespeichert.")
    
    #--------------------------------------------------------
    # Mittelwert und Standardabweichung der Samples berechnen
    #--------------------------------------------------------
    
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    
    Or_samples = np.exp(samples_transformed[:, 2])

    print()
    print("MCMC Ergebnisse:")
    print("-----------------")
    print(f"Omega_m      = {mean[0]:.6f} ± {std[0]:.6f}")
    print(f"Omega_Lambda = {mean[1]:.6f} ± {std[1]:.6f}")
    print(f"Omega_k      = {mean[2]:.6f} ± {std[2]:.6f}")
    print(f"Omega_r      = {np.mean(Or_samples):.8e} ± {np.std(Or_samples):.8e}")
    
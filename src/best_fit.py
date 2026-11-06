import numpy as np 
from scipy.optimize import minimize

from likelihood import negloglike
from likelihood import data
#from student_likelihood import negloglike_student

#nu = 4.0  # Freiheitsgrade der Student-t-Verteilung


# Zielfunktion für Minimize
def objective(params):
    
    try:
        value = negloglike(params)
        #value = negloglike_student(params, nu=nu)
        
    except (ValueError, ZeroDivisionError, FloatingPointError):
        # ungültiger Parametersatz
        return 1e100
    
    if not np.isfinite(value):
        return 1e100
    
    return value    


def constraint_radiation(params):
    Om, Ol, Ok = params
    return 1.0 - Om - Ol - Ok

def find_best_fit(initial_guess):

    print("Starte Optimierung...")
    print()

    z_max = max(row[0] for row in data)
    print("Maximale Rotverschiebung:", z_max)

    bounds = [
        (0.0, None), # Omega_m
        (None, None), # Omega_lambda
        (None, None)  # Omega_k
    ]

    constraints = [
        {
            "type": "ineq",
            "fun": constraint_radiation
        }
    ]

    # Optimierung
    result = minimize(
        objective,
        x0 = initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 5000,
            "ftol": 1e-10,
            "disp": True
           # "maxls": 50
        }
    )

    return result

if __name__ == "__main__":
    
    # Startwert
    initial_guess = [
        0.29,
        0.69,
        0.01
    ]
    
    result = find_best_fit(initial_guess)
    
    # Ergebnisse
    Om, Ol, Ok = result.x
    Or = 1 - Om - Ol - Ok

    print("Optimierung beendet.")
    print()

    print("Erfolg:", result.success)
    print("Meldung:", result.message)
    print()
    #print("Student-t Freiheitsgrade:", nu)
    print("Beste Parameter:")
    print("-----------------")
    print(f"Omega_m      = {Om:.6f}")
    print(f"Omega_Lambda = {Ol:.6f}")
    print(f"Omega_r      = {Or:.6f}")
    print(f"Omega_k      = {Ok:.6f}")
    print()

    start_value = objective(initial_guess)

    print("Vergleich mit Startpunkt:")
    print("-------------------------")
    print(f"-log(L) Startpunkt = {start_value:.6f}")
    print(f"-log(L) Minimum    = {result.fun:.6f}")
    print(f"Verbesserung        = {start_value - result.fun:.6f}")
    print()

    print("Anzahl Iterationen:")
    print(result.nit)
import matplotlib.pyplot as plt 

from read_supernova_data import load_supernova_data


# Daten laden
data = load_supernova_data("data/Supernovadaten.txt")

# Spalten aufteilen
z = [row[1] for row in data]
mu = [row[2] for row in data]
sigma_mu = [row[3] for row in data]


# Hubble-Diagramm
plt.figure(figsize=(9,16))

plt.errorbar(
    z,
    mu,
    yerr=sigma_mu,
    fmt=".",
    markersize = 4,
    alpha = 0.6,
    capsize = 2
)

plt.xlabel("Rotverschiebung z")
plt.ylabel("Distanzmodul µ")
plt.title("Supernova-Hubble-Diagramm")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from permittivite_models import (model_bruggeman_complex, model_maxwell_garnett,
                                  calcul_aire_occupe, eps_or, eps_si, eps0)

periods = np.array([50, 60, 70, 80, 90, 100]) * 1e-9
f_vals  = [calcul_aire_occupe(p) for p in periods]  # f decreases as period grows

x0 = [(eps_or + eps_si).real / 2, (eps_or + eps_si).imag / 2]

bruggeman_vals = []
for f in f_vals:
    sol = root(model_bruggeman_complex, x0=x0, args=(f, eps_or, eps_si))
    bruggeman_vals.append(sol.x[0] + 1j * sol.x[1])

maxwell_vals = [model_maxwell_garnett(f, eps_or, eps_si) for f in f_vals]

# transforme pour avoir la permittivite effective relative
bruggeman_vals_rel = [v / eps0 for v in bruggeman_vals]
maxwell_vals_rel   = [v / eps0 for v in maxwell_vals]

erreur_abs_bruggeman = [np.abs(np.real(curr) - np.real(prev))
                        for prev, curr in zip(bruggeman_vals_rel[:-1], bruggeman_vals_rel[1:])]

erreur_abs_maxwell = [np.abs(np.real(curr) - np.real(prev))
                      for prev, curr in zip(maxwell_vals_rel[:-1], maxwell_vals_rel[1:])]

fig, ax = plt.subplots()

# data
ax.plot(periods[1:], erreur_abs_bruggeman, marker='o', color = "red", label='Bruggeman')
ax.plot(periods[1:], erreur_abs_maxwell,   marker='o',color = "blue", label='Maxwell-Garnett')

# linear fits
coeffs_b = np.polyfit(np.log10(periods[1:]), np.log10(erreur_abs_bruggeman), 1)
coeffs_m = np.polyfit(np.log10(periods[1:]), np.log10(erreur_abs_maxwell), 1)

fit_b = 10**np.polyval(coeffs_b, np.log10(periods[1:]))
fit_m = 10**np.polyval(coeffs_m, np.log10(periods[1:]))

ax.plot(periods[1:], fit_b, '-', color="salmon", label=f'Régression Bruggeman')
ax.plot(periods[1:], fit_m, '-', color="lightblue",  label=f'Régression Maxwell-Garnett')




ax.set_xlabel("Période [m]", fontsize = 15)
ax.set_xscale('log')
ax.set_ylabel("Erreur sur la permittivité effective",fontsize = 15)
ax.set_yscale('log')
ax.legend(fontsize = 12)
plt.tight_layout()
plt.show()
plt.savefig('taille_inclusion_modele_emile_2')


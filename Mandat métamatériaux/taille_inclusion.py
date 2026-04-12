
from fonction_calcul_permittivite_effective import matrices_qui_ne_dependent_pas_du_rayon, permittivite_effective
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse import diags
import scipy.sparse.linalg as spla

Lx = 1.0e-6 
L_eau = 0.5e-6 
eps0 = 8.8542e-12
periods = np.array([50, 60, 70, 80, 90, 100]) * 1e-9 

resultats = [matrices_qui_ne_dependent_pas_du_rayon(p, Lx/2, L_eau/2) for p in periods]


results = [permittivite_effective(300e-9, r) for r in resultats]
eps_eff_x_vals, eps_eff_y_vals, ecart_type_x, ecart_type_y,eps_si_x_rel,eps_si_y_rel = zip(*results)


erreur_abs_x = [np.abs(np.real(ex_curr) - np.real(ex_prev))
                for ex_prev, ex_curr in zip(eps_eff_x_vals[:-1], eps_eff_x_vals[1:])]
erreur_abs_y = [np.abs(np.real(ey_curr) - np.real(ey_prev))
                for ey_prev, ey_curr in zip(eps_eff_y_vals[:-1], eps_eff_y_vals[1:])]

std_erreur_x = [np.sqrt(ecart_type_x[i]**2 + ecart_type_x[i+1]**2
                - 2*np.cov(np.real(eps_si_x_rel[i]).ravel(),
                           np.real(eps_si_x_rel[i+1]).ravel())[0,1])
                for i in range(len(ecart_type_x)-1)]

std_erreur_y = [np.sqrt(ecart_type_y[i]**2 + ecart_type_y[i+1]**2
                - 2*np.cov(np.real(eps_si_y_rel[i]).ravel(),
                           np.real(eps_si_y_rel[i+1]).ravel())[0,1])
                for i in range(len(ecart_type_y)-1)]



fig, ax = plt.subplots()
ax.errorbar(periods[1:], erreur_abs_x, yerr=std_erreur_x,
            marker='o', color="red", capsize=4, label='Erreur en x')
ax.errorbar(periods[1:], erreur_abs_y, yerr=std_erreur_y,
            marker='o', color="blue", capsize=4, label='Erreur en y')
ax.set_xlabel("Période [m]", fontsize=15)
ax.set_xscale('log')
ax.set_ylabel("Erreur sur la permittivité effective", fontsize=15)
ax.set_yscale('log')
ax.legend(fontsize=12)
plt.tight_layout()


plt.show()
plt.savefig('taille_inclusion_emile_2')

# import subprocess
# subprocess.run(["say", "le code a fini de run"])

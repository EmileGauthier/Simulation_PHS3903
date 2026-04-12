
from fonction_calcul_permittivite_effective import matrices_qui_ne_dependent_pas_du_rayon, permittivite_effective
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse import diags
import scipy.sparse.linalg as spla

#i cant go lower than the grid size which is dx= dy = 10 nm, so the period between deux inclusions ne peut pas être plus petit que 
# 10nm.
Lx = 1.0e-6 
L_eau = 0.5e-6 # Épaisseur de la couche d'eau (m)
eps0 = 8.8542e-12
coordonnees = np.linspace(0.1e-6, L_eau - 0.1e-6, 10)
resultats = [matrices_qui_ne_dependent_pas_du_rayon(0.1e-6,  Lx/2, source_y) for source_y in coordonnees]


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
ax.errorbar(0.5e-6 - coordonnees[1:], erreur_abs_x, yerr=std_erreur_x,
            marker='o', color="red", capsize=4, label='Erreur en x')
ax.errorbar(0.5e-6 - coordonnees[1:], erreur_abs_y, yerr=std_erreur_y,
            marker='o', color="blue", capsize=4, label='Erreur en y')
plt.xlabel("Distance du fluorophore de la couche de silicium [m]", fontsize = 15)
plt.xscale('log')
plt.ylabel("Erreur sur la permittivité effective", fontsize = 15)
plt.yscale('log')
plt.legend(fontsize = 12)
plt.tight_layout()
plt.show()
plt.savefig('distance_fluorophore_emile_2')
# import subprocess
# subprocess.run(["say", "I'm done now"])




















































































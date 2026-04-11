
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
eps_eff_x_vals, eps_eff_y_vals = zip(*results)


erreur_abs_x = [np.abs(np.real(ex_curr) - np.real(ex_prev))
                for ex_prev, ex_curr in zip(eps_eff_x_vals[:-1], eps_eff_x_vals[1:])]
erreur_abs_y = [np.abs(np.real(ey_curr) - np.real(ey_prev))
                for ey_prev, ey_curr in zip(eps_eff_y_vals[:-1], eps_eff_y_vals[1:])]

plt.plot(periods[1:], erreur_abs_x, marker='o', color = "red", label='Erreur en x')
plt.plot(periods[1:], erreur_abs_y, marker='o', color = "blue", label='Erreur en y')
plt.xlabel("Période [m]", fontsize = 15)
plt.xscale('log')
plt.ylabel("Erreur absolue", fontsize = 15)
plt.yscale('log')
plt.legend(fontsize = 12)
plt.tight_layout()
plt.show()
# import subprocess
# subprocess.run(["say", "le code a fini de run"])

from fonction_calcul_permittivite_effective import matrices_qui_ne_dependent_pas_du_rayon, permittivite_effective
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse import diags
import scipy.sparse.linalg as spla
import time


# analyse de convergence avec la valeur théorique de la permittivité et la valeur calculée de la permittivité effective
# plus on approxime en moyennant sur plus de noeuds, plus on s'éloigne de la valeur théorique 
# re-définit le rayon comme étant un multiple du maillage, tel que dès que le rayon est plus grand ou égal au maillage, on inclut plus d'un noeud
Lx = 1.0e-6 
L_eau = 0.5e-6 # Épaisseur de la couche d'eau (m)



resultats = matrices_qui_ne_dependent_pas_du_rayon(0.1e-6,Lx/2, L_eau/2)


multiples = [40e-9, 80e-9, 160e-9, 320e-9, 640e-9, 1280e-9, 2560e-9]
results = [permittivite_effective(m, resultats) for m in multiples]
eps_eff_x_vals, eps_eff_y_vals = zip(*results)



erreur_abs_x = [np.abs(np.real(ex_curr) - np.real(ex_prev))
                for ex_prev, ex_curr in zip(eps_eff_x_vals[:-1], eps_eff_x_vals[1:])]
erreur_abs_y = [np.abs(np.real(ey_curr) - np.real(ey_prev))
                for ey_prev, ey_curr in zip(eps_eff_y_vals[:-1], eps_eff_y_vals[1:])]

plt.plot(multiples[1:], erreur_abs_x, marker='o', color = "red", label='Erreur en x')
plt.plot(multiples[1:], erreur_abs_y, marker='o', color = "blue", label='Erreur en y')
plt.xlabel("Valeur du rayon [m]", fontsize = 15)
plt.xscale('log')
plt.ylabel("Erreur absolue", fontsize = 15)
plt.yscale('log')
plt.legend(fontsize = 12)
plt.tight_layout()
plt.show()
# import subprocess
# subprocess.run(["say", "I'm done now"])

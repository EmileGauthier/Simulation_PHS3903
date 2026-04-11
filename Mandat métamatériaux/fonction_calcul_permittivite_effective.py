
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse import diags
import scipy.sparse.linalg as spla
import time

def matrices_qui_ne_dependent_pas_du_rayon(period,x_source, y_source):
    ### ÉTAPE D'INITIALISATION
    ## Paramètres
    # Constantes physiques 
    mu0 = 1.2567e-6 # Perméabilité du vide (H/m)
    eps0 = 8.8542e-12 # Permittivité du vide (F/m)
    c = 2.9979e8 # Vitesse de la lumière dans le vide (m/s)

    # Paramètres physiques
    wavelength_src = 568e-9 # Pic d'émission du Alexa Fluor 555 (m)
    omega_src = 2*np.pi*c/wavelength_src
    eps_eau = eps0 * 1.78 # Permittivité relative de l'eau à 568 nm
    eps_or = eps0 * (14.07 + 0.32j) # Permittivité relative de l'or à 568 nm
    eps_si = eps0 * (16.24 + 0.26j) # Permittivité relative du silicium à 568 nm
    #eps_eau = eps0 * 1 # Permittivité relative de l'eau à 568 nm
    #eps_or = eps0 * 2 # Permittivité relative de l'or à 568 nm
    #eps_si = eps0 * 3 # Permittivité relative du silicium à 568 nm

    # Paramètres géométriques
    L_eau = 0.5e-6 # Épaisseur de la couche d'eau (m)
    L_si = 1.0e-6 # Épaisseur de la couche de silicium (m)
    L_or = 0.5e-6 # Épaisseur de la couche d'or (m)
    Lx = 1.0e-6 # Largeur du domaine de simulation en x (sans PML) (m)
    Ly = L_eau + L_si + L_or # Largeur du domaine de simulation en y (sans PML) (m)

    # Pour les inclusions
    inclusions_bool = True
    Nx_inc = 10 # Nombre selon x
    Ny_inc = 10 # Nombre selon y
    rayon_inclusion = 0.4* period 
    #period = 0.1e-6 # Espacement centre-à-centre

    dx = 0.01e-6 # Pas de discrétisation dans la direction x (m)
    dy = 0.01e-6 # Pas de discrétisation dans la direction y (m)

    Nx=int(np.rint(Lx/dx + 1)) # Nombre de noeuds dans la direction x (excluant les PML)
    Ny=int(np.rint(Ly/dy + 1)) # Nombre de noeuds dans la direction y (excluant les PML)
    N_PML = int(0.8*Ny) #20  # Nombre de noeuds pour les PML
    Lx_PML = N_PML*dx # Épaisseur des régions PML dans la direction x (m)
    Ly_PML = N_PML*dy # Épaisseur des régions PML dans la direction y (m)
    Nx_tot = Nx + 2*N_PML
    Ny_tot = Ny + 2*N_PML
    N = Nx_tot * Ny_tot

    ### Distribution de la permittivité et de la perméabilité
    #eps_mat = sp.sparse.lil_matrix((Nx_tot*Ny_tot,Nx_tot*Ny_tot), dtype=np.complex128) 
    #mu_mat = sp.sparse.lil_matrix((Nx_tot*Ny_tot,Nx_tot*Ny_tot), dtype=np.complex128) 
    eps_diag = np.zeros(N, dtype=np.complex128)
    mu_diag  = np.full(N, mu0, dtype=np.complex128) #matrice N juste avec mu0


    # premier remplacement lil_matrix(N,N) créer une matrice N par N, c'est quand même un gros gaspillage d'espace considérant
    # qu'on n'utilise que la diagonale, même en utilisant sparse, compresed, ça reste bcp de mémoire. au lieu de NxN ça store 3N. tandis que 
    # avec la méthode np.zeros, ce n'est qu'une matrice N. C'est avantageux de garder la définition comme un array de la diagonal parce que toutes les opérations
    # qui seront effectuées sur les valeurs, seront effectuées sur un array plutôt qu'une matrice NxN. Qui plus est, on ne définit pas une matrice NxN et ensuite son inverse, 
    # on ne calcule son inverse qu"une fois à la fin



    n = 5/2
    lnR_0 = 2 * -12
    sigma_max = - n * eps0 * c * lnR_0/(2*Lx_PML)


    #on fait la même chose qu'avant pour définir le grid, les inclusions mais sans nested loops

    # le grid définit par les points i et j, ensuite associé à coordonnées de chaque noeud stocké dans x_arr et y_arr
    i_idx = np.repeat(np.arange(1, Ny_tot + 1), Nx_tot)   # shape (N,)
    j_idx = np.tile(np.arange(1, Nx_tot + 1), Ny_tot)     # shape (N,)
    x_arr = (j_idx - 1) * dx
    y_arr = (i_idx - 1) * dy

    # défini les valeurs de permittivite selon la position
    y_phys = y_arr - Ly_PML
    eps_diag = np.where(y_phys <= L_eau, eps_eau,
               np.where(y_phys <= L_eau + L_si, eps_si, eps_or))

    #distance_from_frontier mais vectorisé
    # initialise des vecteurs pour les distances
    dist_x = np.zeros(N)
    dist_y = np.zeros(N)
    in_PML = np.zeros(N, dtype=bool)

#créer des masques pour identifier chaque noeud dans les PML
    mask_i_lo = i_idx <= N_PML #en bas 
    mask_i_hi = i_idx >= Ny + N_PML # en haut
    mask_j_lo = j_idx <= N_PML # à gauche
    mask_j_hi = j_idx >= Nx + N_PML # à droite

    #défini la distance du début du PML pour attribuer une valeur d'absorption
    dist_y = np.where(mask_i_lo, N_PML - i_idx,
             np.where(mask_i_hi, i_idx - N_PML - Ny, 0))
    dist_x = np.where(mask_j_lo, N_PML - j_idx,
             np.where(mask_j_hi, j_idx - Nx - N_PML, 0))
    in_PML = mask_i_lo | mask_i_hi | mask_j_lo | mask_j_hi

    #trouve sigma pour tous les noeuds dans les PML
    sigma_e = np.where(in_PML,
                       sigma_max * ((dist_x / N_PML)**n + (dist_y / N_PML)**n),
                       0.0)
    sigma_m = mu0 * sigma_e / eps0

    #ajoute les valeurs de eps et mu pour les noeuds dans les PML
    eps_diag = eps_diag.astype(np.complex128)
    eps_diag[in_PML] += 1j * sigma_e[in_PML] / omega_src
    mu_diag[in_PML]  += 1j * sigma_m[in_PML] / omega_src

    # inclusions 
    if inclusions_bool:
        x_centre = dx * N_PML + Lx / 2
        y_centre = dy * N_PML + Ly / 2
        x0 = x_centre - (Nx_inc - 1) / 2 * period
        y0 = y_centre - (Ny_inc - 1) / 2 * period
    #définition des inclusions circulaires et la valeur de eps ajouté à eps dia
        for ix in range(Nx_inc):
            for iy in range(Ny_inc):
                xc = x0 + ix * period
                yc = y0 + iy * period
                mask_inc = (x_arr - xc)**2 + (y_arr - yc)**2 <= rayon_inclusion**2
                eps_diag[mask_inc] = eps_or

    #réorganise la matrice 1D en 2D 
    eps_2D = eps_diag.reshape((Ny_tot, Nx_tot))

    # définit l'inverse de la matric de eps_mat et mu_mat_csc sous forme compressed
    eps_mat_inv = sparse.diags(1.0 / eps_diag, format='csc')
    mu_mat_csc  = sparse.diags(mu_diag, format='csc')

    # tableau des indices des noeuds
    pl = np.arange(N) 


    #changer l'indexation pour que ca match ce avec quoi Python fonctionne
    i_flat = i_idx - 1  
    j_flat = j_idx - 1  

    # défini les dernieres ranges et colonne puisque c'est impossible de faire forward difference
    derniere_row = (i_flat == Ny_tot - 1)
    derniere_col = (j_flat == Nx_tot - 1)

    #Construction de dyf des matrices 1D pour la diagonale et diagonale sup
    #défini les rangé où il y aura des valeurs non nulles
    dyf_rows = np.concatenate([pl, pl[~derniere_row], pl[derniere_row]])
    #défini les colonnes où il y aura des valeurs non nulles
    dyf_cols = np.concatenate([pl,
                                pl[~derniere_row] + Nx_tot,
                                j_flat[derniere_row]]) 
    
    #attribue la valeur de chaque noeud (régulier ou derniere rangée, derniere colonne)
    dyf_data = np.concatenate([-np.ones(N) / dy,
                                np.ones((~derniere_row).sum()) / dy,
                                np.ones(derniere_row.sum()) / dy])
    
    #construction de la matrice avec la valeur et la position dans la matrice
    Dyf = sparse.csc_matrix((dyf_data, (dyf_rows, dyf_cols)), shape=(N, N))

    # meme chose, presque, pour dxf
    dxf_rows = np.concatenate([pl, pl[~derniere_col], pl[derniere_col]])
    dxf_cols = np.concatenate([pl,
                                pl[~derniere_col] + 1,
                                i_flat[derniere_col] * Nx_tot])  
    dxf_data = np.concatenate([-np.ones(N) / dx,
                                np.ones((~derniere_col).sum()) / dx,
                                np.ones(derniere_col.sum()) / dx])

    Dxf = sparse.csc_matrix((dxf_data, (dxf_rows, dxf_cols)), shape=(N, N))

    #calcul Dx Dy backward
    Dxb = -Dxf.T.tocsc()
    Dyb = -Dyf.T.tocsc()

    #définition de A
    A = (Dxb @ eps_mat_inv @ Dxf
       + Dyb @ eps_mat_inv @ Dyf
       + (omega_src**2) * mu_mat_csc)


    #implémentation de la source
    j_source = int(np.rint(x_source/dx + 1)) + N_PML
    i_source = int(np.rint(y_source/dy + 1)) + N_PML
    jx = np.zeros(N)
    jx[(i_source - 1) * Nx_tot + (j_source - 1)] = 1.0

    #définition de b
    b = Dyb @ eps_mat_inv @ jx


    # approximation pour la methode gmres, convergence plus rapide
    M_diag = sparse.diags(1.0 / A.diagonal(), format='csc')
    precond = spla.LinearOperator(A.shape, matvec=lambda x: M_diag @ x)


    # solution avec gmres selon l'approximation donnée, 
    hz, info = spla.gmres(A, b.ravel(), M=precond, atol=1e-8, restart=200, maxiter=1000)
    if info != 0:
        print(f"GMRES warning: info={info}")

    #calcul des champs avec loi de faraday
    Ex_flat = (-1 / (1j * omega_src)) * eps_mat_inv @ (-Dyf @ hz + jx)
    Ey_flat = (-1 / (1j * omega_src)) * eps_mat_inv @ ( Dxf @ hz)

    #reshape les champs selon Ny et Nx
    Hz = hz.reshape((Ny_tot, Nx_tot))
    Ex = Ex_flat.reshape((Ny_tot, Nx_tot))
    Ey = Ey_flat.reshape((Ny_tot, Nx_tot))
    Dx_champ = eps_2D * Ex
    Dy_champ = eps_2D * Ey
   
    return (Ex, Ey, Dx_champ, Dy_champ, eps_2D,
            dx, dy, N_PML, Nx, Ny, L_eau, L_si)





def permittivite_effective(rayon, resultats):
    Ex, Ey, Dx_champ, Dy_champ, eps_2D, dx, dy, N_PML, Nx, Ny, L_eau, L_si = resultats

    #Étape 1 défini quantité de noeuds
    span_y = min(int(rayon / dy), Ny // 2)
    span_x = min(int(rayon / dx), Nx // 2)

    #Étape 2.1 défini la cellule de moyennage
    ky, kx = np.mgrid[-span_y:span_y+1, -span_x:span_x+1] #Définit kx et ky, les array qui définisse un grid carré autour d'un noeud donné
    circle_mask = ((ky * dy)**2 + (kx * dx)**2) <= rayon**2 #Créer un masque circulaire pour ne prendre en compte que les voisins dans un rayon de 300nm
    masque = circle_mask.astype(np.float64) #Definit la quantité de noeuds dans le masque
    masque = masque / masque.sum() #Definit le poids de chaque noeud pour utiliser dans moyenne_masque (étape 2.2)

    #Étape 2.2
    from scipy.ndimage import convolve
    #J'avais pensé faire des nested loops mais ça prend vrm plus de temps et de mémoire, et convolve le fait plus rapidement et avec
    # des vecteurs
    #va faire la moyenne des champs des noeud identifié dans le rayon de la cellule
    def moyenne_masque(champ):
        return (convolve(np.real(champ), masque, mode='nearest')
            + 1j * convolve(np.imag(champ), masque, mode='nearest')) #rajoute la partie réelle et imaginaire manuellement pq convolve ne supporte pas des compelexes
    #[https://numpy.org/devdocs/reference/generated/numpy.convolve.html]

    #Étape 2.3
    #Fait la moyenne pour E et D en x et y 
    Emoy_x = moyenne_masque(Ex)
    Emoy_y = moyenne_masque(Ey)
    Dmoy_x = moyenne_masque(Dx_champ)
    Dmoy_y = moyenne_masque(Dy_champ)


    #Étape 3.1
    #Trouve la moyenne de la permittivité effective pour tous les noeuds
    eps_eff_x = Dmoy_x/Emoy_x
    eps_eff_y = Dmoy_y/Emoy_y


    #Étape 3.2
    #Définit les limites de la matrice eps_eff_si qui est pertinente pour notre analyse dans le silicium
    i_si_start = N_PML + int(L_eau / dy) #décalé à cause des PML
    i_si_end   = N_PML + int((L_eau + L_si) / dy)

    eps_eff_si_x = eps_eff_x[i_si_start:i_si_end, N_PML:N_PML+Nx] #slice en 2D pour aller chercher que la partie de silicium
    eps_eff_si_y = eps_eff_y[i_si_start:i_si_end, N_PML:N_PML+Nx]

    #Étape 4 
    #Fait la moyenne des moyennes de permittivité effective
    mean_eps_eff_x = np.mean(eps_eff_si_x)
    mean_eps_eff_y = np.mean(eps_eff_si_y)

    return mean_eps_eff_x, mean_eps_eff_y




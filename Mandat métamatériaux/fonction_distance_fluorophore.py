
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse import diags
import scipy.sparse.linalg as spla
import time


def matrices_qui_ne_dependent_pas_du_rayon():
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
    r = 0.04e-6 # Rayon des inclusions
    Nx_inc = 10 # Nombre selon x
    Ny_inc = 10 # Nombre selon y
    period = 0.1e-6 # Espacement centre-à-centre

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

    # pas besoin pour le calcul de la permittivité
    # def distance_from_frontier(i,j,Ny,Nx,N_PML):
    #     dist_x = 0
    #     dist_y = 0
    #     in_PML = False
    #     if i <= N_PML:
    #         in_PML = True
    #         dist_y = N_PML - i
    #     if i >= Ny + N_PML:
    #         in_PML = True
    #         dist_y = i - N_PML - Ny
    #     if j <= N_PML:
    #         in_PML = True
    #         dist_x = N_PML-j
    #     if j >= Nx + N_PML:
    #         in_PML = True
    #         dist_x = j - Nx - N_PML
    #     return dist_x, dist_y, in_PML

    n = 5/2
    lnR_0 = 2 * -12
    sigma_max = - n * eps0 * c * lnR_0/(2*Lx_PML)

    #définiton de la grid avec les matériau et PML et inclusions
    # for i in np.arange(1,Ny_tot+1,1):
    #     y=(i-1)*dy
    #     for j in np.arange(1,Nx_tot+1,1):
    #         x=(j-1)*dx
    #         pl = (i-1)*Nx_tot + j

    #         # Sans PML
    #         mu_mat[pl-1,pl-1] = mu0
            
    #         if y - Ly_PML <= L_eau:
    #             eps_mat[pl-1,pl-1] = eps_eau
    #         elif y - Ly_PML >= L_eau and y - Ly_PML <= (L_eau + L_si):
    #             eps_mat[pl-1,pl-1] = eps_si
    #         else:
    #             eps_mat[pl-1,pl-1] = eps_or

    #         # Ajout des PML
    #         dist_x, dist_y, in_PML = distance_from_frontier(i,j,Ny,Nx,N_PML)

    #         if in_PML == True:
    #             sigma_e = sigma_max * ((dist_x/N_PML)**n + (dist_y/N_PML)**n)
    #             sigma_m = mu0 * sigma_e / eps0

    #             eps_mat[pl-1,pl-1] += 1j * sigma_e/omega_src
    #             mu_mat[pl-1,pl-1] += 1j * sigma_m/omega_src

    #         # Inclusions circulaires++
    #         if inclusions_bool == True:
    #             x_center = dx*N_PML + Lx/2
    #             y_center = dy*N_PML + Ly/2

    #             x0 = x_center - (Nx_inc - 1) / 2 * period
    #             y0 = y_center - (Ny_inc - 1) / 2 * period

    #             for ix in range(Nx_inc):
    #                 for iy in range(Ny_inc):
    #                     xc = x0 + ix * period
    #                     yc = y0 + iy * period
    #                     if (x - xc)**2 + (y - yc)**2 <= r**2:
    #                         eps_mat[pl-1,pl-1] = eps_or
    #                         break  # Un seul match suffit
    #                 else:
    #                     continue
    #                 break  # Sortir des deux boucles dès qu'une inclusion est trouvée

    #on fait la même chose qu'avant pour définir le grid, les inclusions mais sans nested loops

    # le grid définit par les points i et j, ensuite associé à coordonnées de chaque noeud stocké dans x_arr et y_arr
    i_idx = np.repeat(np.arange(1, Ny_tot + 1), Nx_tot)   # shape (N,)
    j_idx = np.tile(np.arange(1, Nx_tot + 1), Ny_tot)     # shape (N,)
    x_arr = (j_idx - 1) * dx
    y_arr = (i_idx - 1) * dy

    # défini les valeurs de epsilon selon la position
    y_phys = y_arr - Ly_PML
    eps_diag = np.where(y_phys <= L_eau, eps_eau,
               np.where(y_phys <= L_eau + L_si, eps_si, eps_or))

    # initialise des vecteurs pour les distances
    dist_x = np.zeros(N)
    dist_y = np.zeros(N)
    in_PML = np.zeros(N, dtype=bool)

#créer des masques pour identifier chaque partie qu'on observe
    mask_i_lo = i_idx <= N_PML #en bas 
    mask_i_hi = i_idx >= Ny + N_PML # en haut
    mask_j_lo = j_idx <= N_PML # à gauche
    mask_j_hi = j_idx >= Nx + N_PML # à droite

    dist_y = np.where(mask_i_lo, N_PML - i_idx,
             np.where(mask_i_hi, i_idx - N_PML - Ny, 0))
    dist_x = np.where(mask_j_lo, N_PML - j_idx,
             np.where(mask_j_hi, j_idx - Nx - N_PML, 0))
    in_PML = mask_i_lo | mask_i_hi | mask_j_lo | mask_j_hi

    #trouve sigma pour tous les noeuds d'un
    sigma_e = np.where(in_PML,
                       sigma_max * ((dist_x / N_PML)**n + (dist_y / N_PML)**n),
                       0.0)
    sigma_m = mu0 * sigma_e / eps0

    eps_diag = eps_diag.astype(np.complex128)
    eps_diag[in_PML] += 1j * sigma_e[in_PML] / omega_src
    mu_diag[in_PML]  += 1j * sigma_m[in_PML] / omega_src

    # Circular inclusions (vectorised)
    if inclusions_bool:
        x_center = dx * N_PML + Lx / 2
        y_center = dy * N_PML + Ly / 2
        x0 = x_center - (Nx_inc - 1) / 2 * period
        y0 = y_center - (Ny_inc - 1) / 2 * period

        for ix in range(Nx_inc):
            for iy in range(Ny_inc):
                xc = x0 + ix * period
                yc = y0 + iy * period
                mask_inc = (x_arr - xc)**2 + (y_arr - yc)**2 <= r**2
                eps_diag[mask_inc] = eps_or

    eps_2D = eps_diag.reshape((Ny_tot, Nx_tot))



    # Extraire la diagonale (valeurs de permittivité)
    #eps_diag = eps_mat.diagonal()

    # Remettre en forme 2D
    #eps_2D = eps_diag.reshape((Ny_tot, Nx_tot))

    # définit l'inverse de la matric de eps_mat et mu_mat_csc sous forme compressed
    eps_mat_inv = sparse.diags(1.0 / eps_diag, format='csc')
    mu_mat_csc  = sparse.diags(mu_diag, format='csc')

    #avant : pl = (i-1)*Nx_tot + j
    pl = np.arange(N) 


    # # Plot partie réelle
    # plt.figure()
    # plt.imshow(np.real(eps_2D[N_PML:Ny+N_PML,N_PML:Nx+N_PML])/eps0, origin='lower', aspect='auto')
    # plt.colorbar(label='Re(ε)')
    # plt.title("Permittivité (partie réelle)")
    # plt.xlabel("x")
    # plt.ylabel("y")

    # # Plot partie imaginaire (PML)
    # plt.figure()
    # plt.imshow(np.imag(eps_2D)/eps0, origin='lower', aspect='auto')
    # plt.colorbar(label='Im(ε)')
    # plt.title("Permittivité (partie imaginaire - PML)")
    # plt.xlabel("x")
    # plt.ylabel("y")

    #plt.show()


    # eps_diag = eps_mat.diagonal()
    # eps_mat_inv = sp.sparse.diags(1/eps_diag)

    i_flat = i_idx - 1  # 0-based row
    j_flat = j_idx - 1  # 0-based col

    last_row = (i_flat == Ny_tot - 1)
    last_col = (j_flat == Nx_tot - 1)

    # Dyf rows
    dyf_rows = np.concatenate([pl, pl[~last_row], pl[last_row]])
    dyf_cols = np.concatenate([pl,
                                pl[~last_row] + Nx_tot,
                                j_flat[last_row]])  # wraps to row 0
    dyf_data = np.concatenate([-np.ones(N) / dy,
                                np.ones((~last_row).sum()) / dy,
                                np.ones(last_row.sum()) / dy])

    Dyf = sparse.csc_matrix((dyf_data, (dyf_rows, dyf_cols)), shape=(N, N))

    # Dxf rows
    dxf_rows = np.concatenate([pl, pl[~last_col], pl[last_col]])
    dxf_cols = np.concatenate([pl,
                                pl[~last_col] + 1,
                                i_flat[last_col] * Nx_tot])  # wraps to col 0
    dxf_data = np.concatenate([-np.ones(N) / dx,
                                np.ones((~last_col).sum()) / dx,
                                np.ones(last_col.sum()) / dx])

    Dxf = sparse.csc_matrix((dxf_data, (dxf_rows, dxf_cols)), shape=(N, N))

    Dxb = -Dxf.T.tocsc()
    Dyb = -Dyf.T.tocsc()

    # --- Assemble A (curl-curl) ---
    A = (Dxb @ eps_mat_inv @ Dxf
       + Dyb @ eps_mat_inv @ Dyf
       + (omega_src**2) * mu_mat_csc)


    # Implémentation de la source

    x_source = Lx/2
    y_source = L_eau/2

    j_source = int(np.rint(x_source/dx + 1)) + N_PML
    i_source = int(np.rint(y_source/dy + 1)) + N_PML

    # jx = np.zeros((Nx_tot*Ny_tot,1),dtype=np.double)
    # jy = np.zeros((Nx_tot*Ny_tot,1),dtype=np.double)
    # jx[(i_source-1)*Nx_tot+(j_source-1)] = 1

    jx = np.zeros(N)
    jx[(i_source - 1) * Nx_tot + (j_source - 1)] = 1.0

    
    b = Dyb @ eps_mat_inv @ jx

    # lu = spla.splu(A)
    # hz = lu.solve(b)

    # 1. Build preconditioner
    M_diag = sparse.diags(1.0 / A.diagonal(), format='csc')
    precond = spla.LinearOperator(A.shape, matvec=lambda x: M_diag @ x)

    # print(f"A.shape = {A.shape}")
    # print(f"b.ravel().shape = {b.ravel().shape}")
    # print(f"N = {N}")
    # print(f"Nx_tot={Nx_tot}, Ny_tot={Ny_tot}")
    # print(f"N_PML={N_PML}, Nx={Nx}, Ny={Ny}")


    # 2. Solve
    hz, info = spla.gmres(A, b.ravel(), M=precond, atol=1e-8, restart=200, maxiter=1000)
    if info != 0:
        print(f"GMRES warning: info={info}")

    Ex_flat = (-1 / (1j * omega_src)) * eps_mat_inv @ (-Dyf @ hz + jx)
    Ey_flat = (-1 / (1j * omega_src)) * eps_mat_inv @ ( Dxf @ hz)


    Hz = hz.reshape((Ny_tot, Nx_tot))
    Ex = Ex_flat.reshape((Ny_tot, Nx_tot))
    Ey = Ey_flat.reshape((Ny_tot, Nx_tot))
    Dx_champ = eps_2D * Ex
    Dy_champ = eps_2D * Ey
   

    return (Ex, Ey, Dx_champ, Dy_champ, eps_2D,
            dx, dy, N_PML, Nx, Ny, L_eau, L_si)





    # Formatage des matrices de dérivées

    # Dxf = sp.sparse.lil_matrix((Nx_tot*Ny_tot,Nx_tot*Ny_tot), dtype=np.double) 
    # Dyf = sp.sparse.lil_matrix((Nx_tot*Ny_tot,Nx_tot*Ny_tot), dtype=np.double)  

    # for i in np.arange(1,Ny_tot+1,1):
    #     for j in np.arange(1,Nx_tot+1,1):
    #         # remplir la ligne pl de la matrice M
    #         pl = (i-1)*Nx_tot + j

    #         if i == Ny_tot:
    #             pc = pl; Dyf[pl-1,pc-1] = -1/dy
    #             pc = j; Dyf[pl-1,pc-1] = 1/dy
    #         elif j == Nx_tot:
    #             pc = pl; Dxf[pl-1,pc-1] = -1/dx
    #             pc = (i-1)*Nx_tot + 1; Dxf[pl-1,pc-1] = 1/dx
    #         else:
    #             pc = pl; Dyf[pl-1,pc-1] = -1/dy; Dxf[pl-1,pc-1] = -1/dx
    #             pc = (i)*Nx_tot+j; Dyf[pl-1,pc-1] = 1/dy
    #             pc = (i-1)*Nx_tot+(j+1); Dxf[pl-1,pc-1] = 1/dx

    # Dxb = -Dxf.T
    # Dyb = -Dyf.T

    # ### ÉTAPE DE RÉSOLUTION NUMÉRIQUE

    # # Conversion en CSC pour efficacité
    # Dxf_csc = Dxf.tocsc()
    # Dyf_csc = Dyf.tocsc()
    # Dxb_csc = Dxb.tocsc()
    # Dyb_csc = Dyb.tocsc()
    # eps_mat_inv_csc = eps_mat_inv.tocsc()
    # mu_mat_csc = mu_mat.tocsc()

    # # Assemblage de la matrice A (curl-curl centré)
    # A = Dxb_csc @ eps_mat_inv_csc @ Dxf_csc \
    # + Dyb_csc @ eps_mat_inv_csc @ Dyf_csc \
    # + (omega_src**2) * mu_mat_csc

    # # Assemblage du vecteur source b
    # b = Dyb_csc @ eps_mat_inv_csc @ jx \
    # - Dxb_csc @ eps_mat_inv_csc @ jy

    # # Résolution par décomposition LU
    # lu = spla.splu(A)
    # hz = lu.solve(b)

    # # Récupération des champs Ex et Ey via la loi de Faraday
    # ex = (-1/(1j*omega_src)) * eps_mat_inv_csc @ (-Dyf_csc @ hz + jx)
    # ey = (-1/(1j*omega_src)) * eps_mat_inv_csc @ ( Dxf_csc @ hz + jy)

    # # Remise en forme 2D
    # Hz = hz.reshape((Ny_tot, Nx_tot))
    # Ex = ex.reshape((Ny_tot, Nx_tot))
    # Ey = ey.reshape((Ny_tot, Nx_tot))



    # ### VISUALISATION
    # # Hz - partie réelle dans la zone physique (sans PML)
    # plt.figure()
    # plt.imshow(np.real(Hz[N_PML:Ny+N_PML, N_PML:Nx+N_PML]),
    #            cmap='RdBu', origin='lower', aspect='auto')
    # plt.colorbar(label='Re(Hz)')
    # plt.title("Champ Hz (partie réelle)")
    # plt.xlabel("x")
    # plt.ylabel("y")

    # # Hz - amplitude log dans tout le domaine (avec PML)
    # plt.figure()
    # plt.imshow(np.log10(np.abs(Hz) + 1e-30),
    #            cmap='viridis', origin='lower', aspect='auto')
    # plt.colorbar(label='log|Hz|')
    # plt.title("Champ Hz (amplitude log, domaine complet)")
    # plt.xlabel("x")
    # plt.ylabel("y")

    # # Ex - partie réelle dans la zone physique
    # plt.figure()
    # plt.imshow(np.real(Ex[N_PML:Ny+N_PML, N_PML:Nx+N_PML]),
    #            cmap='RdBu', origin='lower', aspect='auto')
    # plt.colorbar(label='Re(Ex)')
    # plt.title("Champ Ex (partie réelle)")
    # plt.xlabel("x")
    # plt.ylabel("y")

    # # Ey - partie réelle dans la zone physique
    # plt.figure()
    # plt.imshow(np.real(Ey[N_PML:Ny+N_PML, N_PML:Nx+N_PML]),
    #            cmap='RdBu', origin='lower', aspect='auto')
    # plt.colorbar(label='Re(Ey)')
    # plt.title("Champ Ey (partie réelle)")
    # plt.xlabel("x")
    # plt.ylabel("y")

    # # Intensité |E|² dans la zone physique
    # E_intensity = np.abs(Ex)**2 + np.abs(Ey)**2
    # plt.figure()
    # plt.imshow(np.log10(E_intensity[N_PML:Ny+N_PML, N_PML:Nx+N_PML] + 1e-30),
    #            cmap='hot', origin='lower', aspect='auto')
    # plt.colorbar(label='log|E|²')
    # plt.title("Intensité du champ électrique log|E|²")
    # plt.xlabel("x")
    # plt.ylabel("y")

    # plt.tight_layout()
    # plt.show()



def permittivite_effective(rayon, resultats):
    Ex, Ey, Dx_champ, Dy_champ, eps_2D, dx, dy, N_PML, Nx, Ny, L_eau, L_si = resultats

    #Étape 1
    #Pour tous les noeuds, trouver E_x, E_y and D_x and D_y en utilisant la relation D = premittivité * E
    # Dx_champ = eps_2D * Ex
    # Dy_champ = eps_2D * Ey

    #Étape 2.1
    #Créer un masque de rayon inférieur à la longueur d'onde qui se propage, pour cibler tous les noeuds dans le masque
    #rayon = 300e-9
    #span_y = int(rayon/dy) #Définit combien de noeuds en y et en x on inclut dans la moyenne
    #span_x = int(rayon/dx) 


    span_y = min(int(rayon / dy), Ny // 2)
    span_x = min(int(rayon / dx), Nx // 2)


    ky, kx = np.mgrid[-span_y:span_y+1, -span_x:span_x+1] #Définit kx et ky, les array qui définisse un grid carré autour d'un noeud donné
    circle_mask = ((ky * dy)**2 + (kx * dx)**2) <= rayon**2 #Créer un masque circulaire pour ne prendre en compte que les voisins dans un rayon de 300nm
    masque = circle_mask.astype(np.float64) #Definit la quantité de noeuds dans le masque
    masque = masque / masque.sum() #Definit le poids de chaque noeud pour utiliser dans moyenne_masque (étape 2.2)

    #Étape 2.2
    from scipy.ndimage import convolve
    #J'avais pensé faire des nested loops mais ça prend vrm plus de temps et de mémoire, et convolve le fait plus rapidement et avec
    # des vecteurs
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

    #eps_eff = (Dmoy_x*Emoy_x + Dmoy_y*Emoy_y)/(Emoy_y**2 + Emoy_x**2) #Pour préserver la partie complexe pour pouvoir mieux comparer, pq ça revient à la même chose

    #Étape 3.2
    #Définit les limites de la matrice eps_eff_si qui est pertinente pour notre analyse dans le silicium
    i_si_start = N_PML + int(L_eau / dy) #décalé à cause des PML
    i_si_end   = N_PML + int((L_eau + L_si) / dy)

    eps_eff_si_x = eps_eff_x[i_si_start:i_si_end, N_PML:N_PML+Nx] #slice en 2D pour aller chercher que la partie de silicium
    eps_eff_si_y = eps_eff_y[i_si_start:i_si_end, N_PML:N_PML+Nx]

    #Étape 4

    # from numpy import linalg as LA
    # norme_x = LA.norm(mean_eps_eff_x)
    # norme_y = LA.norm(mean_eps_eff_y)
    # real_mean = np.sqrt(norme_x**2 + norme_y**2)

    #Fait la moyenne des moyennes de permittivité effective
    mean_eps_eff_x = np.mean(eps_eff_si_x)
    mean_eps_eff_y = np.mean(eps_eff_si_y)

    #i need to preserve the real and imaginary part


    # print(mean_eps_eff_x, mean_eps_eff_y)
    # print(np.mean([mean_eps_eff_y+ mean_eps_eff_x]))
    # print(eps_or,eps_si)
    return mean_eps_eff_x, mean_eps_eff_y




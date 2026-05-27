import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg, eigs, spsolve, bicgstab
from math import prod as list_prod
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import sys
import time
import os



import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix






def A_fixed_implicit(Ns, discretisation_steps, D, r, ijk2n_map):
    
    # THIS FUNCTION CREATES THE MATRIX A_FIXED THAT CONTAINS THE DISCRETISATION OF THE TERMS
    #                   U^{n+1} - dt DELTA D(z) U^{n+1} - dt r(z) U^{n+1}


    ######## INPUT: ######
    ## Ns contains the number of points of the discretisation:
    ### If the spatial domain is 1D, Ns = [Nx, Nz, N] where N = Nx * Nz
    ### If the spatial domain is 2D, Ns = [Nx, Ny, Nz, N] where N = Nx * Ny * Nz

    ## discretisation_steps contains the constants from the phenotypic-spatial discretisation:
    ### If the spatial domain is 1D, discretisation_steps = [dt, DtDx2, DtDz2, dz] 
    ### If the spatial domain is 2D, discretisation_steps = [dt, DtDx2, DtDy2, DtDz2, dz] 

    ## D is the vector of spatial diffusion coefficients defined over the discretised phenotypic space
    ## Dz is a diffusion coefficient in the phenotype space (generally very small or zero)

    ## ijk2n_map is the map between the 2 or 3 variables, namely from (x, z) in 2D or (x, y, z) in 3D to the single index n
    
    ## r is the proliferation rate 

    dimensions = len(Ns) - 1
    N = Ns[-1]
    A = lil_matrix((N, N))

    if dimensions == 2:
        
        Nx, Nz = Ns[:-1]
        dt, DtDx2 = discretisation_steps[:-1]
        
        for k in range(Nz):
                for i in range(Nx):
                    idx = ijk2n_map[i, k] #n(i,j,k, Nx, Ny) # 1D index for (i,j,k)
                    element = DtDx2 * D[k]
                    
                    A[idx, idx] =  1 + 2 * element - dt*r     # Central coefficient
                    
                    # X-direction
                    if i > 0:
                        A[idx, ijk2n_map[i-1, k]] -= element # Left neighbor x-Dx
                    else:
                        A[idx, ijk2n_map[i+1, k]] -= element # Neumann BC at x = 0
                    if i < Nx - 1:
                        A[idx, ijk2n_map[i+1, k]] -= element # Right neighbor x+Dx
                    else:
                        A[idx, ijk2n_map[i-1, k]] -= element # Neumann BC at x = Nx

    elif dimensions == 3:

        Nx, Ny, Nz  = Ns[:-1]
        dt, DtDx2, DtDy2 = discretisation_steps[:-1]
        
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    idx = ijk2n_map[i, j, k] #n(i,j,k, Nx, Ny) # 1D index for (i,j,k)
                    element_x = DtDx2 * D[k]
                    element_y = DtDy2 * D[k]
                    
                    A[idx, idx] =  1 + 2 * element_x + 2 * element_y - dt*r     # Central coefficient
                    
                    # X-direction
                    if i > 0:
                        A[idx, ijk2n_map[i-1, j, k]] -= element_x # Left neighbor x-Dx
                    else:
                        A[idx, ijk2n_map[i+1, j, k]] -= element_x # Neumann BC at x = 0
                    if i < Nx - 1:
                        A[idx, ijk2n_map[i+1, j, k]] -= element_x # Right neighbor x+Dx
                    else:
                        A[idx, ijk2n_map[i-1, j, k]] -= element_x # Neumann BC at x = Nx

                    # Y-direction
                    if j > 0:
                        A[idx, ijk2n_map[i, j-1, k]] -= element_y  # Back neighbor y-Dy
                    else:
                        A[idx, ijk2n_map[i, j+1, k]] -= element_y  # Neumann BC at y = 0
                    if j < Ny - 1:
                        A[idx, ijk2n_map[i, j+1, k]] -= element_y # Forward neighbor y+Dy
                    else:
                        A[idx, ijk2n_map[i, j-1, k]] -= element_y  # Neumann BC at y = Ny
                    
    return A.tocsr()


def A_fixed_implicit_mutation(Ns, discretisation_steps, M, theta, ijk2n_map):

    # THIS FUNCTION CREATES THE MATRIX A_FIXED THAT CONTAINS THE DISCRETISATION OF THE TERMS
    #                   dt * integral [M(z) U^{n+1}] dz - dt * U^{n+1}

    ######## INPUT: ######
    ## Ns contains the number of points of the discretisation:
    ### If the spatial domain is 1D, Ns = [Nx, Nz, N] where N = Nx * Nz
    ### If the spatial domain is 2D, Ns = [Nx, Ny, Nz, N] where N = Nx * Ny * Nz

    ## discretisation_steps contains the constants from the phenotypic-spatial discretisation:
    ### If the spatial domain is 1D, discretisation_steps = [dt, DtDx2, DtDz2, dz] 
    ### If the spatial domain is 2D, discretisation_steps = [dt, DtDx2, DtDy2, DtDz2, dz] 

    ## M is the vector of the phenotypic mutation kernel evaluated on the discretised phenotypic space
    ## theta is a mutation coefficient 

    ## ijk2n_map is the map between the 2 or 3 variables, namely from (x, z) in 2D or (x, y, z) in 3D to the single index n
    

    N = Ns[-1]
    dimensions = len(Ns) - 1
    A = lil_matrix((N, N))

    if dimensions == 2:

        Nx, Nz = Ns[:-1]
        dt = discretisation_steps[0]
        dz = discretisation_steps[-1]
        
        for i in range(Nx):
            for k in range(Nz):
                idx = ijk2n_map[i, k]
                A[idx, idx] += dt * theta

                element = dt * (1./3) * dz * theta * M[k]
                
                idx_p = ijk2n_map[i, 0]
                A[idx, idx_p] -= element
                
                idx_p = ijk2n_map[i, Nz-1]
                A[idx, idx_p] -= element
                
                for kp in range(1, Nz, 2):
                    idx_p = ijk2n_map[i, kp]
                    A[idx, idx_p] -= 4 * element
                    
                for kp in range(2, Nz, 2):
                    idx_p = ijk2n_map[i, kp]
                    A[idx, idx_p] -= 2 * element


    elif dimensions == 3:

        Nx, Ny, Nz = Ns[:-1]
        dt = discretisation_steps[0]
        dz = discretisation_steps[-1]
        
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    idx = ijk2n_map[i, j, k]
                    A[idx, idx] += dt * theta

                    element = dt * (1./3) * dz * theta * M[k]
                    
                    idx_p = ijk2n_map[i, j, 0]
                    A[idx, idx_p] -= element
                
                    idx_p = ijk2n_map[i, j, Nz-1]
                    A[idx, idx_p] -= element

                    for kp in range(1, Nz, 2):
                        idx_p = ijk2n_map[i, j, kp]
                        A[idx, idx_p] -= 4 * element
                    
                    for kp in range(2, Nz, 2):
                        idx_p = ijk2n_map[i, j, kp]
                        A[idx, idx_p] -= 2 * element

    return A.tocsr()



def A_Picard_cached_implicit_idx_parallel(intK_cache, J_term_cache, Ns, discretisation_steps, ijk2n_map, n2ijk_map):
    N = Ns[-1]
    dimensions = len(Ns) - 1

    # Ensure arrays are correct dtypes and contiguous (best practice)
    # n2ijk_map should be shape (N, dimensions) with dtype int32
    n2ijk_map = np.asarray(n2ijk_map, dtype=np.int32, order='C')
    ijk2n_map = np.asarray(ijk2n_map, dtype=np.int32, order='C')
    J_term_cache = np.asarray(J_term_cache, dtype=np.float64, order='C')
    intK_cache = np.asarray(intK_cache, dtype=np.float64, order='C')
    discretisation_steps = np.asarray(discretisation_steps, dtype=np.float64, order='C')
    Ns = np.asarray(Ns, dtype=np.int32)

    if dimensions == 2:
        rows, cols, vals, counts = build_entries_2d(N, Ns, discretisation_steps, n2ijk_map, ijk2n_map, J_term_cache, intK_cache)
    elif dimensions == 3:
        rows, cols, vals, counts = build_entries_3d(N, Ns, discretisation_steps, n2ijk_map, ijk2n_map, J_term_cache, intK_cache)
    else:
        raise ValueError("Only 2D or 3D supported")

    nnz = np.sum(counts)
    rows = rows[:nnz].astype(np.int32)
    cols = cols[:nnz].astype(np.int32)
    vals = vals[:nnz].astype(np.float64)

    A = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return A




def assembly_fixed_matrices(Ns, discretisation_steps, D1, r1, D2, r2, M1, theta1, M2, theta2, ijk2n_map):

    A1_0 = A_fixed_implicit(Ns, discretisation_steps, D1, r1, ijk2n_map)   # THE FIXED PART FOR U1
    A2_0 = A_fixed_implicit(Ns, discretisation_steps, D2, r2, ijk2n_map)   # THE FIXED PART FOR U2

    A1_M = A_fixed_implicit_mutation(Ns, discretisation_steps, M1, theta1, ijk2n_map)
    A2_M = A_fixed_implicit_mutation(Ns, discretisation_steps, M2, theta2, ijk2n_map)

    A1_fixed = A1_0 + A1_M
    A2_fixed = A2_0 + A2_M

    return A1_fixed, A2_fixed



@njit(parallel=True)
def build_entries_2d(N, Ns, discretisation_steps, n2ijk_map, ijk2n_map, J_term_cache, intK_cache):
    max_entries_per_row = 3  # diag + 2 neighbors (with BC handling)
    rows = np.empty(N * max_entries_per_row, dtype=np.int32)
    cols = np.empty(N * max_entries_per_row, dtype=np.int32)
    vals = np.empty(N * max_entries_per_row, dtype=np.float64)
    counts = np.zeros(N, dtype=np.int32)

    dt = discretisation_steps[0]
    DtDx2 = discretisation_steps[1]
    Nx = Ns[0]

    for idx in prange(N):
        # n2ijk_map is expected as shape (N, 2): [i, k]
        i = n2ijk_map[idx, 0]
        k = n2ijk_map[idx, 1]

        # diagonal
        pos = idx * max_entries_per_row + counts[idx]
        rows[pos] = idx
        cols[pos] = idx
        vals[pos] = 2.0 * DtDx2 * J_term_cache[i, k] + dt * intK_cache[i, k]
        counts[idx] += 1

        # left neighbor (or Neumann mirror)
        if i > 0:
            ni = ijk2n_map[i-1, k]
            pos = idx * max_entries_per_row + counts[idx]
            rows[pos] = idx
            cols[pos] = ni
            vals[pos] = -DtDx2 * J_term_cache[i-1, k]
            counts[idx] += 1
        else:
            ni = ijk2n_map[i+1, k]
            pos = idx * max_entries_per_row + counts[idx]
            rows[pos] = idx
            cols[pos] = ni
            vals[pos] = -DtDx2 * J_term_cache[i+1, k]
            counts[idx] += 1

        # right neighbor (or Neumann mirror)
        if i < Nx - 1:
            ni = ijk2n_map[i+1, k]
            pos = idx * max_entries_per_row + counts[idx]
            rows[pos] = idx
            cols[pos] = ni
            vals[pos] = -DtDx2 * J_term_cache[i+1, k]
            counts[idx] += 1
        else:
            ni = ijk2n_map[i-1, k]
            pos = idx * max_entries_per_row + counts[idx]
            rows[pos] = idx
            cols[pos] = ni
            vals[pos] = -DtDx2 * J_term_cache[i-1, k]
            counts[idx] += 1

    return rows, cols, vals, counts


@njit(parallel=True)
def build_entries_3d(N, Ns, discretisation_steps, n2ijk_map, ijk2n_map, J_term_cache, intK_cache):
    print('check if this version does actually work fine!')
    max_entries_per_row = 5  # diag + 4 neighbors (x and y) (z not coupled here)
    rows = np.empty(N * max_entries_per_row, dtype=np.int32)
    cols = np.empty(N * max_entries_per_row, dtype=np.int32)
    vals = np.empty(N * max_entries_per_row, dtype=np.float64)
    counts = np.zeros(N, dtype=np.int32)

    dt = discretisation_steps[0]
    DtDx2 = discretisation_steps[1]
    DtDy2 = discretisation_steps[2]
    Nx = Ns[0]
    Ny = Ns[1]

    for idx in prange(N):
        # n2ijk_map expected as shape (N, 3): [i, j, k]
        i = n2ijk_map[idx, 0]
        j = n2ijk_map[idx, 1]
        k = n2ijk_map[idx, 2]

        # diagonal
        pos = idx * max_entries_per_row + counts[idx]
        rows[pos] = idx
        cols[pos] = idx
        vals[pos] = 2.0 * DtDx2 * J_term_cache[i, j, k] + 2.0 * DtDy2 * J_term_cache[i, j, k] + dt * intK_cache[i, j, k]
        counts[idx] += 1

        # X-direction neighbors
        if i > 0:
            ni = ijk2n_map[i-1, j, k]
            pos = idx * max_entries_per_row + counts[idx]
            rows[pos] = idx
            cols[pos] = ni
            vals[pos] = -DtDx2 * J_term_cache[i-1, j, k]
            counts[idx] += 1
        else:
            ni = ijk2n_map[i+1, j, k]
            pos = idx * max_entries_per_row + counts[idx]
            rows[pos] = idx
            cols[pos] = ni
            vals[pos] = -DtDx2 * J_term_cache[i+1, j, k]
            counts[idx] += 1

        if i < Nx - 1:
            ni = ijk2n_map[i+1, j, k]
            pos = idx * max_entries_per_row + counts[idx]
            rows[pos] = idx
            cols[pos] = ni
            vals[pos] = -DtDx2 * J_term_cache[i+1, j, k]
            counts[idx] += 1
        else:
            ni = ijk2n_map[i-1, j, k]
            pos = idx * max_entries_per_row + counts[idx]
            rows[pos] = idx
            cols[pos] = ni
            vals[pos] = -DtDx2 * J_term_cache[i-1, j, k]
            counts[idx] += 1

        # Y-direction neighbors
        if j > 0:
            nj = ijk2n_map[i, j-1, k]
            pos = idx * max_entries_per_row + counts[idx]
            rows[pos] = idx
            cols[pos] = nj
            vals[pos] = -DtDy2 * J_term_cache[i, j-1, k]
            counts[idx] += 1
        else:
            nj = ijk2n_map[i, j+1, k]
            pos = idx * max_entries_per_row + counts[idx]
            rows[pos] = idx
            cols[pos] = nj
            vals[pos] = -DtDy2 * J_term_cache[i, j+1, k]
            counts[idx] += 1

        if j < Ny - 1:
            nj = ijk2n_map[i, j+1, k]
            pos = idx * max_entries_per_row + counts[idx]
            rows[pos] = idx
            cols[pos] = nj
            vals[pos] = -DtDy2 * J_term_cache[i, j+1, k]
            counts[idx] += 1
        else:
            nj = ijk2n_map[i, j-1, k]
            pos = idx * max_entries_per_row + counts[idx]
            rows[pos] = idx
            cols[pos] = nj
            vals[pos] = -DtDy2 * J_term_cache[i, j-1, k]
            counts[idx] += 1

    return rows, cols, vals, counts



def calculate_rhos(U1, U2, dz, Ns, ijk2n_map):
    
    # in input: 
    # U1, U2 = 1xN arrays of phenotypic densities
    # dz = discretisation steps
    # Ns = [Nx, Nz, Nx*Nz] in 1D or [Nx, Ny, Nz, Nx*Ny*Nz] in 2D spatial domains
    # ijk2n_map = Nx*Nz array in 1D or Nx*Ny*Nz array in 2D whose elements contain 
    # the global index, as evaluated in index_n
    
    # in output:
    # rho1, rho2 = densities of U1, U2, obtained by integrating in phenotype (using Simpson rule)

    
    if len(Ns) == 3:  # Case 1: 2D grid (i, k, z)
        Nz = Ns[1]

        # Gather indices once
        idx = ijk2n_map[:, :Nz]   # shape (Ns[0], Nz)
        U1_vals = U1[idx]         # shape (Ns[0], Nz)
        U2_vals = U2[idx]

        # Simpson rule integration along z
        I1 = U1_vals[:, 0] + U1_vals[:, -1] + 4*U1_vals[:, 1:-1:2].sum(axis=1) + 2*U1_vals[:, 2:-1:2].sum(axis=1)
        I2 = U2_vals[:, 0] + U2_vals[:, -1] + 4*U2_vals[:, 1:-1:2].sum(axis=1) + 2*U2_vals[:, 2:-1:2].sum(axis=1)
        
        return dz*I1/3, dz*I2/3

    elif len(Ns) == 4:  # Case 2: 3D grid (i, j, k, z)
        Nz = Ns[2]

        idx = ijk2n_map[:, :, :Nz]   # shape (Ns[0], Ns[1], Nz)
        U1_vals = U1[idx]            # shape (Ns[0], Ns[1], Nz)
        U2_vals = U2[idx]

        # Simpson rule integration along z
        I1 = U1_vals[:, :, 0] + U1_vals[:, :, -1] + 4*U1_vals[:, :, 1:-1:2].sum(axis=2) + 2*U1_vals[:, :, 2:-1:2].sum(axis=2)
        I2 = U2_vals[:, :, 0] + U2_vals[:, :, -1] + 4*U2_vals[:, :, 1:-1:2].sum(axis=2) + 2*U2_vals[:, :, 2:-1:2].sum(axis=2)

        return dz*I1/3, dz*I2/3

    else:
        raise ValueError("Ns must have length 3 or 4.")


def check_convergence_iterative_solver(conv1, conv2, picard_iter, t):

    if conv1 > 0 and conv2 >0:
        print('t = ', t)
        print('convs = ', conv1, '  ', conv2)
        print('picard iterations = ', picard_iter)
        sys.exit('The solver could not solve the system')

    return


def competition_kernel(z, choice, M, Mp):
    
    # in input:
    # z = discretised phenotypic space (1xNz array)
    # choice = value to indicate what type of competition kernel to select (see below)
    # M, Mp = phenotype switching kernels
    
    # in output: 
    # K = kernel evaluated in z
    # weighted_integral_K = the weighted integral <K>  
    
    if choice == 0:
        K = np.ones((len(z), len(z)))
        
    elif choice == 1:
        K = (z[:, None] - z[None,:])**2
        
    elif choice == 2:
        K = 1 - (z[:,None] - z[None,:])**2
        
    elif choice == 3:
        K = (-4 * z * (z - 1))[:, None] * np.ones((1, len(z)))

    elif choice == 4:
        K = (z**2)[:, None] * np.ones((1, len(z)))

    elif choice == 5:
        K = (1 - (z - 1)**2 )[:, None] * np.ones((1, len(z)))

    elif choice == 6:
        K = z[:, None] * z[None, :]

    else:
        raise ValueError('The choice for the phenotypic kernel is not clear. Please reselect it or add into the list of possible functions in \033[93m competition_kernel \033[00m')

    weighted_integral_K = double_integration_simpson(K, M, Mp, z[1]-z[0], z[1]-z[0])

    return K, weighted_integral_K


def compute_integral_terms_vectorised(U1_picard, U2_picard, K1, K2, dz, Ns, ijk2n_map):
    """
    Compute intK_cache for 2D (Nx,Nz) or 3D (Nx,Ny,Nz) spatial domains.
    """
    Nz = Ns[-2]
    intK_vals = kernel_K(U1_picard, U2_picard, K1, K2, dz, Nz, ijk2n_map)

    # Reshape back into spatial dimensions
    if len(Ns) == 3:   # 2D: Nx × Nz
        Nx, Nz = Ns[0:2]
        intK_cache = intK_vals.T.reshape(Nx, Nz)

    elif len(Ns) == 4: # 3D: Nx × Ny × Nz
        Nx, Ny, Nz = Ns[0:3]
        intK_cache = intK_vals.T.reshape(Nx, Ny, Nz)

    else:
        raise ValueError("Ns must describe a 2D or 3D domain")

    return intK_cache



def diffusion_functions(z, parameter_list):

    diffusion_terms = ['D1', 'alpha11', 'alpha12', 'D2', 'alpha21', 'alpha22']
    D_and_alpha_functions = []
    
    for i in range(len(parameter_list)):
        
        P = parameter_list[i]     # it contains 4 elements, the last one, P[3], is a flag

        if P[3] == 0:  # CONSTANT
            diffusion_function = P[0]*np.ones(len(z))

        elif P[3] == 1: # SIGMOIDAL
            diffusion_function = P[0] + P[1]/( 1 + np.exp(-P[2]*(z-0.5))) 

        elif P[3] == 2: # PARABOLIC
            diffusion_function = P[0] + P[1]*z + P[2]*z**2

        else:
            raise ValueError('The choice for the diffusion function is not clear. Please reselect it or add into the list of possible functions in \033[93m diffusion_functions \033[00m')
        
        D_and_alpha_functions.append(diffusion_function)

    return D_and_alpha_functions



def double_integration_simpson(K, M, Mp, dz, dzp):
    Nzp = len(Mp)
    I1 = K[:, 0] * Mp[0] + K[:, Nzp-1] * Mp[Nzp-1]
    for kp in range(1,Nzp-1):
        if kp % 2 == 1:
            I1 += 4*K[:, kp]* Mp[kp]
        else:
            I1 += 2*K[:, kp]* Mp[kp]
    I1 = dzp*I1/3
    
    Nz = len(M)
    I2 = I1[0] * M[0] + I1[Nz-1] * M[Nz-1] 
    for k in range(1,Nz-1):
        if k % 2 == 1:
            I2 += 4 * I1[k]* M[k]
        else:
            I2 += 2 * I1[k]* M[k]
    I2 = dz*I2 /3
    return I2



def evaluate_discretisation_steps(increments):
    
    # in input:
    # increments = [dt, dx, dz]
    # increments = [dt, dx, dy, dz]
    
    # in output: 
    # discretisation_steps = [dt, DtDx2, DtDz2, dz] 
    # discretisation_steps = [dt, DtDx2, DtDy2, DtDz2, dz] 

    dimensions = len(increments) - 1

    if dimensions == 2:
        dt, dx, dz = increments
        DtDx2 = dt/(dx**2)
        discretisation_steps = [dt, DtDx2, dz]
    
    elif dimensions == 3:
        dt, dx, dy, dz = increments
        DtDx2 = dt/(dx**2)
        DtDy2 = dt/(dy**2)
        discretisation_steps = [dt, DtDx2, DtDy2, dz] 
    
    return discretisation_steps



def evaluate_kernels_and_functions(z, diffusion_parameter_list, beta_pars, K_kernel_choices, M_pars):

    # in input:
    # z = discretised phenotypic space (1xNz array)
    # diffusion_parameter_list = parameters for the diffusion functions D_i, alpha_ij (see diffusion_functions)
    # beta_pars = array containing the competition coefficients beta_ij
    # K_kernel_choices = array containing the choices for competition kernels (see competition_kernel)
    # M_pars = [M_pars[0], M_pars[1]], where M_pars[i] contains the parameters for M_i

    # it returns:
    # a list D_and_alpha_functions containing all the evaluated D_i and alpha_ij functions evaluated in z
    # the competition kernels betaKij evaluated evaluated over (z, z)
    # the two phenotypic switching kernels M_i evaluated in z
    

    D_and_alpha_functions = diffusion_functions(z, diffusion_parameter_list)

    M1 = phenotype_switching_kernel(z, M_pars[0])
    M2 = phenotype_switching_kernel(z, M_pars[1])
    
    dz = z[1] - z[0]

    beta11, beta12, beta21, beta22 = beta_pars
    kernel_choice_11, kernel_choice_12, kernel_choice_21, kernel_choice_22 = K_kernel_choices

    betaK11 = beta11 * competition_kernel(z, kernel_choice_11, M1, M1)[0]
    betaK22 = beta22 * competition_kernel(z, kernel_choice_22, M2, M2)[0]
    betaK12 = beta12 * competition_kernel(z, kernel_choice_12, M1, M2)[0]
    betaK21 = beta21 * competition_kernel(z, kernel_choice_21, M1, M2)[0]

    return D_and_alpha_functions, betaK11, betaK12, betaK21, betaK22, M1, M2


def evaluate_weighted_integrals(z, D_and_alpha_functions, betaK11, betaK12, betaK21, betaK22, beta_pars, M1, M2):
    
    dz = z[1]-z[0]
    
    weighted_diffusion_coefficients = []

    for i in [0, 1, 2]:    
        weighted_diffusion_coefficients.append( Simpson(D_and_alpha_functions[i]*M1,dz) )
    for i in [3, 4, 5]:    
        weighted_diffusion_coefficients.append( Simpson(D_and_alpha_functions[i]*M2,dz) )

    
    beta11, beta12, beta21, beta22 = beta_pars

    mean_K11 = double_integration_simpson(betaK11, M1, M1, dz, dz)/beta11
    mean_K12 = double_integration_simpson(betaK12, M1, M2, dz, dz)/beta12
    mean_K21 = double_integration_simpson(betaK21, M2, M1, dz, dz)/beta21
    mean_K22 = double_integration_simpson(betaK22, M2, M2, dz, dz)/beta22

    weighted_competition_coefficients = [mean_K11, mean_K12, mean_K21, mean_K22]
    
    
    return weighted_diffusion_coefficients, weighted_competition_coefficients
    
def index_n(Ns):
    
    # This function assigns a unique index to the couple ik in 1D 
    # or to the triplet ijk in 2D spatial domains
    
    # in input 
    # Ns = [Nx, Nz, Nx*Nz] in 1D or [Nx, Ny, Nz, Nx*Ny*Nz] in 2D spatial domains
    
    # in output
    # idx2n_map = Nx*Nz matrix in 1D or Nx*Ny*Nz in 2D that contains the global index 

    if len(Ns) == 3: 
        Nx, Nz = Ns[:-1]    
        idx2n_map = np.zeros((Nx, Nz), dtype=np.int32)
        for k in range(Nz):
            for i in range(Nx):
                idx2n_map[i,k] = k*Nx + i

    elif len(Ns) == 4:
        Nx, Ny, Nz = Ns[:-1]    
        idx2n_map = np.zeros((Nx, Ny, Nz), dtype=np.int32)
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    idx2n_map[i,j,k] = k*Nx*Ny + j*Nx + i
                    
    else:
        sys.exit('ERROR: check dimensions of the phenotypic-spatial domain')

    return idx2n_map


def indices_ijk(Ns):
    
    # This function returns the couple ik in 1D or the triplet ijk in 2D spatial domains
    # associated to the unique index evaluated by index_n
    
    
    # in input 
    # Ns = [Nx, Nz, Nx*Nz] in 1D or [Nx, Ny, Nz, Nx*Ny*Nz] in 2D spatial domains
    
    # in output
    # n2ijk_map = list whose elements are ordered elements [i,k] for 1D 
    # and [i,j,k] for 2D spatial domains
    
    N = Ns[-1]
    Nx = Ns[0]
    
    n2ijk_map = []
    
    for idx in range(N):
        if len(Ns) == 3:
            k = idx // Nx
            i = idx % Nx
            n2ijk_map.append([i,k])
        
        elif len(Ns) == 4:
            Ny = Ns[1]
            k = idx // (Nx * Ny)
            j = (idx % (Nx * Ny)) // Nx
            i = idx % Nx
            n2ijk_map.append([i,j,k])

    return n2ijk_map


def int_with_kern_K_vectorised(U, K, dz, Nz, ijk2n_map):
    """
    Compute Simpson rule integral:
        ∫ K(z, s) * U(x, y, s) ds
    for all spatial coordinates at once.

    U : (Ntotal,) array
    K : (Nz_out, Nz) kernel matrix
    dz : float
    Nz : int
    ijk2n_map : array mapping spatial coords -> U index
                shape (Nx, Nz)   for 2D
                shape (Nx, Ny, Nz) for 3D
    """

    # Gather U values for all spatial points
    U_vals = U[ijk2n_map].reshape(-1, Nz)  # shape (M, Nz)

    # --- Sanity checks ---
    if K.ndim != 2:
        raise ValueError(f"K must be 2D, got shape {K.shape}")
    if U_vals.shape[1] != Nz:
        raise ValueError(f"U_vals second dim {U_vals.shape[1]} != Nz={Nz}")
    if K.shape[1] != Nz:
        raise ValueError(f"K second dim {K.shape[1]} != Nz={Nz}")

    # --- Simpson integration ---
    I = (
        K[:, 0][:, None] * U_vals[:, 0][None, :] +
         K[:, -1][:, None] * U_vals[:, -1][None, :]
        + 4*K[:, 1:-1:2] @ U_vals[:, 1:-1:2].T
        + 2*K[:, 2:-1:2] @ U_vals[:, 2:-1:2].T
    )  # shape (Nz_out, M)

    return dz * I / 3 # (Nz_out, M)




def J_term(rho1, rho2, alpha1, alpha2, Ns):
    
    if len(Ns) == 3:
        J = rho1[:, None] * alpha1[None, :] + rho2[:, None] * alpha2[None, :]
    elif len(Ns) == 4:
        J = rho1[:, :, None] * alpha1[None, None, :] + rho2[:, :, None] * alpha2[None, None, :]
    else:
        raise ValueError("Ns must have length 3 or 4.")
    return J


def kernel_K(U1, U2, K1, K2, dz, Nz, ijk2n_map):
    I1 = int_with_kern_K_vectorised(U1, K1, dz, Nz, ijk2n_map)
    I2 = int_with_kern_K_vectorised(U2, K2, dz, Nz, ijk2n_map)
    return I1 + I2  # shape (Nz_out, M)


def mesh_refinement(original_discretisation, variable, refs_x, refs_z):
    Np = len(original_discretisation)
    if variable == 'x' or variable == 'y':
        Np_refined = Np + refs_x
    else:
        Np_refined = Np + refs_z
    refined_discretisation = np.linspace(original_discretisation[0], original_discretisation[-1], Np_refined)
    return Np_refined, refined_discretisation


def phenotype_switching_kernel(z, M_pars): # phenotype-switching kernel

    # in input:
    # z = discretised phenotypic space (1xNz array)
    # M_pars = [mu_i1, sigma2_i1, mu_i2 (optional), sigma2_i2 (optional), lambda_i (optional)] 
    # mu_ij = the position of the peak j
    # sigma2_i2 = "variance" relative to peak j
    # lambda_i = balancing parameter between the two peaks
    
    # it returns a Gaussian function of the sum of 2 Gaussian functions, with integral = 1 over z

    if len(M_pars) == 2:                    # single peak
        mu, sigma2 = M_pars
        G = np.exp(-(z-mu)**2/sigma2)       

    else:                                   # double peak
        mu_1, sigma2_1, mu_2, sigma2_2, lambda_ = M_pars
        G1 = np.exp(-(z-mu_1)**2/sigma2_1)       
        G2 = np.exp(-(z-mu_2)**2/sigma2_2)       
        G = (1 - lambda_) * G1 + lambda_ * G2
        
    integral_G = Simpson(G, z[1] - z[0])  # Integral for normalise G
    return G / integral_G


def plot_system_functions(z, D1, alpha11, alpha12, D2, alpha21, alpha22, M1, M2, output_folder):
    
    # I store the plot of all functions (diffusion and phenotype switching kernels) in input into the output folder
    
    fig, axs = plt.subplots(2, 4, figsize=(15, 8))  # 2 rows, 3 columns
    axs[0, 0].plot(z, D1, label=r"$D_1(z)$")
    axs[0, 0].legend()
    
    axs[0, 1].plot(z, alpha11, label=r"$\alpha_{11}(z)$")
    axs[0, 1].legend()
    
    axs[0, 2].plot(z, alpha12, label=r"$\alpha_{12}(z)$")
    axs[0, 2].legend()

    axs[0, 3].plot(z, M1, label=r"$M_{1}(z)$")
    axs[0, 3].legend()
    
    axs[1, 0].plot(z, D2, label=r"$D_2(z)$")
    axs[1, 0].legend()
    
    axs[1, 1].plot(z, alpha21, label=r"$\alpha_{21}(z)$")
    axs[1, 1].legend()
    
    axs[1, 2].plot(z, alpha22, label=r"$\alpha_{22}(z)$")
    axs[1, 2].legend()
    
    axs[1, 3].plot(z, M2, label=r"$M_{2}(z)$")
    axs[1, 3].legend()    

    plt.savefig(output_folder + 'diffusion-mutation-functions.png', dpi=300, bbox_inches='tight')

    return 1




def refine1d(U1, U2, original_discretisation, variable, Ns, ijk2n_map, refs_x, refs_z):
    
    # variable = 'x', 'y', 'z'
    if len(U1) != Ns[-1] or len(U2) != Ns[-1]:
        sys.exit('The vector U1 or U2 has not size N! - check inputs to refine1d function!')
    
    
    dims = len(Ns) - 1
    Ns_refined = Ns.copy()
    N_refined, refined_discretisation = mesh_refinement(original_discretisation, variable, refs_x, refs_z)

    if dims == 3:
        Nx, Ny, Nz = Ns[0:-1]
    elif dims == 2:
        Nx, Nz = Ns[0:-1]
    
    if variable == 'x':
        Ns_refined[0] = N_refined
        Ns_refined[-1] = list_prod(Ns_refined[:-1])
        ijk2n_map_refined = index_n(Ns_refined) 
        U1_refined = np.zeros(Ns_refined[-1])
        U2_refined = np.zeros(Ns_refined[-1])

        if dims == 2:
            for k in range(Nz):
                U1_refined[ijk2n_map_refined[:, k]] = np.interp(refined_discretisation, original_discretisation, U1[ijk2n_map[:, k]])
                U2_refined[ijk2n_map_refined[:, k]] = np.interp(refined_discretisation, original_discretisation, U2[ijk2n_map[:, k]])
        
        elif dims == 3:
            for j in range(Ny):
                for k in range(Nz):
                    U1_refined[ijk2n_map_refined[:, j, k]] = np.interp(refined_discretisation, original_discretisation, U1[ijk2n_map[:, j, k]])
                    U2_refined[ijk2n_map_refined[:, j, k]] = np.interp(refined_discretisation, original_discretisation, U2[ijk2n_map[:, j, k]])
        
        
    elif variable == 'y':
        if dims == 2: 
            print('\n ERROR! THE VARIABLE Y DOES NOT EXIST!\n')
            return
        else:
            Ns_refined[1] = N_refined
            Ns_refined[-1] = list_prod(Ns_refined[:-1])
            ijk2n_map_refined = index_n(Ns_refined) 
            U1_refined = np.zeros(Ns_refined[-1])
            U2_refined = np.zeros(Ns_refined[-1])

            for i in range(Nx):
                for k in range(Nz):
                    U1_refined[ijk2n_map_refined[i, :, k]] = np.interp(refined_discretisation, original_discretisation, U1[ijk2n_map[i, :, k]])
                    U2_refined[ijk2n_map_refined[i, :, k]] = np.interp(refined_discretisation, original_discretisation, U2[ijk2n_map[i, :, k]])
            
    elif variable == 'z':
        Ns_refined[-2] = N_refined
        Ns_refined[-1] = list_prod(Ns_refined[:-1])
        ijk2n_map_refined = index_n(Ns_refined) 
        U1_refined = np.zeros(Ns_refined[-1])
        U2_refined = np.zeros(Ns_refined[-1])
    
        if dims == 2:
            for i in range(Nx):
                U1_refined[ijk2n_map_refined[i, :]] = np.interp(refined_discretisation, original_discretisation, U1[ijk2n_map[i, :]])
                U2_refined[ijk2n_map_refined[i, :]] = np.interp(refined_discretisation, original_discretisation, U2[ijk2n_map[i, :]])
        
        elif dims == 3:
            for i in range(Nx):
                for j in range(Ny):
                    U1_refined[ijk2n_map_refined[i, j, :]] = np.interp(refined_discretisation, original_discretisation, U1[ijk2n_map[i, j, :]])
                    U2_refined[ijk2n_map_refined[i, j, :]] = np.interp(refined_discretisation, original_discretisation, U2[ijk2n_map[i, j, :]])
    
    else: 
        print('\n No variable for refinement has been selected\n')
        return 
    
      
    
    return refined_discretisation, Ns_refined, ijk2n_map_refined, U1_refined, U2_refined



def Simpson(G,dz):
    return dz * ( G[0] + G[-1] + 4*np.sum(G[1:-1:2]) + 2*np.sum(G[2:-1:2]))/3


def smoothly_perturbed_initial_condition(U1_star, U2_star, Ns, ijk2n_map, amplitude, period, x, y=None):
    
    print('U1_star = ', U1_star)
    print('U2_star = ', U2_star)
    
    N = Ns[-1]
    
    U1_0 = np.zeros(N)
    U2_0 = np.zeros(N)

    if len(Ns) == 4:
        Nx, Ny, Nz = Ns[:-1]
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    U1_0[ijk2n_map[i,j,k]] += U1_star * (1 + amplitude*np.cos(period*x[i])*np.cos(period*y[j]))
                    U2_0[ijk2n_map[i,j,k]] += U2_star * (1 + amplitude*np.sin(period*x[i])*np.sin(period*y[j]))
                    
    elif len(Ns) == 3:
        Nx, Nz = Ns[:-1]
        for i in range(Nx):
            for k in range(Nz):
                U1_0[ijk2n_map[i,k]] = U1_star * ( 1 + amplitude*np.cos(period*x[i]) )
                U2_0[ijk2n_map[i,k]] = U2_star * ( 1 + amplitude*np.sin(period*x[i]) )
        
    return U1_0, U2_0



def steady_state(r1, r2, weighted_competition_coefficients, beta_pars):
    mean_K11, mean_K12, mean_K21, mean_K22 = weighted_competition_coefficients
    beta11, beta12, beta21, beta22 = beta_pars

    den = beta11*mean_K11*beta22*mean_K22 - beta12*mean_K12*beta21*mean_K21
    num1 = r1*beta22*mean_K22 - r2*beta12*mean_K12
    num2 = r2*beta11*mean_K11 - r1*beta21*mean_K21
    
    return num1/den, num2/den



def store_refinement_data(dt, Ns, t, adaptive_steps, output_folder):
    file_name = output_folder + 'refinement' + str(adaptive_steps) + '_t=' + str(round(t,3)) + '_data.txt'
    with open(file_name, 'w') as f:
        f.write(f"dt = {dt}\n")
        f.write(f"Ns = {Ns[:-1]}\n")
        f.write(f"N = {Ns[-1]}\n")
        f.write(f"t = {t}\n")
        f.write(f"adaptive_steps = {adaptive_steps}\n")
        
        
        
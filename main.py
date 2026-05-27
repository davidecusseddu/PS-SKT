import numpy as np
from functions import *
from parameters import *
# from functions_optimised import *

print('All parameters have been imported... \n The simulation is starting')


export_pars(output_folder)




# initialisation
adaptive_steps = 0
t = 0
previous_picard_iter = 0
picards = []
np_exporting_time = dt*np_export_step




while adaptive_steps <= max_adaptive_steps and t < T:

    if adaptive_steps == 0:

        # Ns is defined in the parameters file
        
        # DEFINISCO LE MAPPE DI INDICI
        ijk2n_map = index_n(Ns)                 # DALLA TRIPLETTA (i,j,k) AL SINGOLO n

        if dimensions == 2:
            Nx, Nz = Ns[:-1]
        elif dimensions == 3:
            Nx, Ny, Nz = Ns[:-1]

        # DISCRETISATION OF THE DOMAIN  OMEGA = [-pi,pi]x[-pi,pi], Z in [0,1]
        x = np.linspace(-np.pi,np.pi,Nx)
        z = np.linspace(0,1,Nz)
        if dimensions == 3:
            y = np.linspace(-np.pi,np.pi,Ny)

    else:
        print('Refinement ' + str(adaptive_steps) + ' in progress... at time t = ' + str(round(t, 3)))

        # Refine in t
        dt = dt*refs_t

        # Refine in x
        x, Ns, ijk2n_map, U1_n, U2_n = refine1d(U1_n, U2_n, x, 'x', Ns, ijk2n_map, refs_x, refs_z)

        if dimensions == 3:
            # Refine in y
            y, Ns, ijk2n_map, U1_n, U2_n = refine1d(U1_n, U2_n, y, 'y', Ns, ijk2n_map, refs_x, refs_z)

        # Refine in z
        z, Ns, ijk2n_map, U1_n, U2_n = refine1d(U1_n, U2_n, z, 'z', Ns, ijk2n_map, refs_x, refs_z)


    store_refinement_data(dt, Ns, t, adaptive_steps, output_folder)

    dx = x[1] - x[0]                    # DISCRETISATION STEP
    dz = z[1] - z[0]
    
    if dimensions == 2:
        discretisation_steps = evaluate_discretisation_steps([dt, dx, dz])
    elif dimensions == 3:
        dy = y[1] - y[0]                # DISCRETISATION STEP
        discretisation_steps = evaluate_discretisation_steps([dt, dx, dy, dz])
        

    n2ijk_map = indices_ijk(Ns)             # DAL SINGOLO n ALLA TRIPLETTA (i,j,k)
    D_and_alpha_functions, betaK11, betaK12, betaK21, betaK22, M1, M2 = evaluate_kernels_and_functions(z, diffusion_parameter_list, beta_pars, K_kernel_choices, M_pars)
    
    D1 = D_and_alpha_functions[0]
    alpha11 = D_and_alpha_functions[1]
    alpha12 = D_and_alpha_functions[2]
    D2 = D_and_alpha_functions[3]
    alpha21 = D_and_alpha_functions[4]
    alpha22 = D_and_alpha_functions[5]

    weighted_diffusion_coefficients, weighted_competition_coefficients = evaluate_weighted_integrals(z, D_and_alpha_functions, betaK11, betaK12, betaK21, betaK22, beta_pars, M1, M2)

    if t == 0:
        
        print("\n Starting the simulation: \n")
        
        # SETTING THE INITIAL CONDITION AND EXPORT IT TO FILE
        #U1_0, U2_0 = initial_condition(U1_star, U2_star, Ns, ijk2n_map, M1, M2)
        #U1_0, U2_0 = add_perturbation(U1_0, U2_0, max_perturbation1, max_perturbation2)

        U1_star, U2_star = steady_state(r1, r2, weighted_competition_coefficients, beta_pars)


        U1_0, U2_0 = smoothly_perturbed_initial_condition(U1_star, U2_star, Ns, ijk2n_map, max_perturbation, perturbation_period, x, y=None)
        U1_n = U1_0[:]
        U2_n = U2_0[:]
        
        
        rho1, rho2 = calculate_rhos(U1_n, U2_n, dz, Ns, ijk2n_map)
        
        plot_system_functions(z, D1, alpha11, alpha12, D2, alpha21, alpha22, M1, M2, output_folder)

    
    # I START ASSEMBLYING THE FINITE DIFFERENCE SYSTEM
    # THE SCHEME IS 100% IMPLICIT. THEREFORE IT TAKES THE FORM A1 U1^{n} = U1^{n-1} AND   A2 U2^{n} = U2^{n-1}

    # MATRICES OF THE COEFFICIENTS
    A1_fixed, A2_fixed = assembly_fixed_matrices(Ns, discretisation_steps, D1, r1, D2, r2, M1, theta1, M2, theta2, ijk2n_map)
    
    refinement = adaptive_steps

    while t <= T and refinement == adaptive_steps:
        
        picard_iter = 0
        Up1 = U1_n.copy()
        Up2 = U2_n.copy()
        
        while picard_iter == 0 or ( (np.linalg.norm(Up1-U1)/max(np.linalg.norm(Up1),1e-12) > picard_tol or np.linalg.norm(Up2-U2)/max(np.linalg.norm(Up2),1e-12) > picard_tol) and picard_iter < picard_max_iter):
        #while (np.linalg.norm(Up1-U1) > picard_tol or np.linalg.norm(Up2-U2) > picard_tol) and picard_iter < picard_max_iter:

            if picard_iter > 0:
                Up1 = U1.copy()
                Up2 = U2.copy()

            rhop1, rhop2 = calculate_rhos(Up1, Up2, dz, Ns, ijk2n_map)    

            
            intK1_cache = compute_integral_terms_vectorised(Up1, Up2, betaK11, betaK12, dz, Ns, ijk2n_map)
            J1_term_cache = J_term(rhop1, rhop2, alpha11, alpha12, Ns)
            
            A1_p = A_Picard_cached_implicit_idx_parallel(intK1_cache, J1_term_cache, Ns, discretisation_steps, ijk2n_map, n2ijk_map)
            A1 = A1_fixed + A1_p
            U1, conv1 = bicgstab(A1, U1_n, x0=Up1, rtol = solver_tol)  # Use Conjugate Gradient method

            
            intK2_cache = compute_integral_terms_vectorised(Up1, Up2, betaK21, betaK22, dz, Ns, ijk2n_map)
            J2_term_cache = J_term(rhop1, rhop2, alpha21, alpha22, Ns)
            
            A2_p = A_Picard_cached_implicit_idx_parallel(intK2_cache, J2_term_cache, Ns, discretisation_steps, ijk2n_map, n2ijk_map)
            A2 = A2_fixed + A2_p
            U2, conv2 = bicgstab(A2, U2_n, x0=Up2, rtol = solver_tol)
                                 
            check_convergence_iterative_solver(conv1, conv2, picard_iter, t)
            
            picard_iter += 1
        
        if picard_iter >= picard_max_iter and picard_max_iter>1: # PICARD DID NOT CONVERGE => REFINEMENT OF THE MESH
            print('relative norm(Up1-U1) = ', np.linalg.norm(Up1-U1)/max(np.linalg.norm(Up1),1e-12))
            print('relative norm(Up2-U2) = ', np.linalg.norm(Up2-U2)/max(np.linalg.norm(Up2),1e-12) )
            print('... Picard iterate did not converge within the maximal iterations, we will now run an adaptive step ... ')
            
            adaptive_steps += 1
            
        else:
            # UPDATE PREVIOUS SOLUTION 
            U1_n = U1[:]
            U2_n = U2[:]
            t += dt
            if t >= np_exporting_time: # AND STORE THE SOLUTION
                print('The solution was exported at time t = ', t)
                np.savez(output_folder + 'U1_t='+str(round(t,3)), U1=U1, t=t)
                np.savez(output_folder + 'U2_t='+str(round(t,3)), U2=U2, t=t)
                np_exporting_time += dt*np_export_step
                rho1, rho2 = calculate_rhos(U1, U2, dz, Ns, ijk2n_map)
                
                
            
        if picard_iter != previous_picard_iter or picard_iter == picard_max_iter :
            # KEEP TRACK OF THE NUMBER OF PICARD ITERATIONS AT EVERY TIME
            picards.append(picard_iter)
            previous_picard_iter = picard_iter
        
        

if adaptive_steps >= max_adaptive_steps:
    print('The number of refinments has reached its maximal value')
else:
    print('The solver has successfully completed the task')



if not os.path.exists(output_folder + 'U1_t=' + str(round(t, 3)) + '.npz'):
    np.savez(output_folder + 'U1_t='+str(round(t,3)), U1=U1, t=t)
    np.savez(output_folder + 'U2_t='+str(round(t,3)), U2=U2, t=t)


np.save(output_folder + 'picards_iterations', picards)


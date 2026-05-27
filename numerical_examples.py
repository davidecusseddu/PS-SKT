
import numpy as np

numerical_example = '1B'

#### DIFFUSION FUNCTIONS ####
# P = [delta, gamma, eta, function_choice]
# function_choice = 0 => constant delta
# function_choice = 1 => sigmoidal
# function_choice = 2 => parabolic


#### PHENOTYPIC KERNELS ####
# K_kernel_choices = choices for [K11, K12, K21, K22]
# K_kernel_choices = 0 => constant = 1
# K_kernel_choices = 1 => (yi-yj)^2
# K_kernel_choices = 2 => 1 - (yi-yj)^2
# K_kernel_choices = 3 => -4*yi*(yi-1)  (constant wrt yj)
# K_kernel_choices = 4 => yi^2          (constant wrt yj)
# K_kernel_choices = 5 => 1 - (yi-1)^2  (constant wrt yj)
# K_kernel_choices = 6 => yi*yj




if numerical_example == '1A' or numerical_example == '1B': 

    # Diffusion functions 
     # for each diffusion function, I create a vector P with all the parameters, 
     # where P[3] acts as a flag for selecting the function definition:
     # P[3] = 0 -> constant:  P[0] 
     # P[3] = 1 -> sigmoidal: P[0] + P[1]/( 1 + np.exp(-P[2]*(z-0.5)))
     # P[3] = 2 -> parabolic: P[0] + P[1]*z + P[2]*z**2 
    
    D1 = [0.001, 0,  0,  0]
    D2 = [0.01,  0,  0,  0]
    alpha11 = [0.0002, 0, 0, 0]
    alpha12 = [0.02, 0, 0, 0]
    alpha21 = [0, 0.3, 30, 1]
    alpha22 = [0.0003, 0, 0, 0]

    # Growth rates:
    r1 = 0.5
    r2 = 0.3
    
    # Competition kernels
    K_kernel_choices = np.array([1, 0, 3, 0])  # choices for [K11, K12, K21, K22]
    beta_pars = [6, 0.06, 0.002, 0.05]         # values for [beta11, beta12, beta21, beta22]

    # Phenotype switching kernels
    M1_pars = [0.5, 0.01]
    if numerical_example == '1A': 
        M2_pars = [0.45, 0.01]
    elif numerical_example == '1B':      
        M2_pars = [0.50, 0.01]
        

elif numerical_example == '2A' or numerical_example == '2B': 
   
    D1 = [0.01, 0,  0,  0]
    D2 = [0.09,  0,  0,  0]
    alpha11 = [0, 0, 0, 0]
    alpha12 = [0, 0.06, -0.03, 2]
    alpha21 = [0.5, 0, -0.5, 2]
    alpha22 = [0, 0, 0, 0]
    
    # Growth rates:
    r1 = 0.5
    r2 = 0.3
    
    # Competition kernels
    if numerical_example == '2A': 
        K_kernel_choices = np.array([1, 4, 5, 2])
    elif numerical_example == '2B':      
        K_kernel_choices = np.array([2, 4, 5, 2])
        
    beta_pars = [125.002, 6.923, 0.027, 0.714]         # values for [beta11, beta12, beta21, beta22]
    
    # Phenotype switching kernels
    M1_pars = [0.5, 0.02]
    M2_pars = [0.5, 0.02]    
    

elif numerical_example == '3A' or numerical_example == '3B':    

    D1 = [0, 0.04, 20,  1]
    D2 = [0.02, 0,  0,  0]
    alpha11 = [0, 0.06, 20, 1]
    alpha12 = [0.00002, -0.00002, 20, 1]
    alpha21 = [8, -8, -20, 1]
    alpha22 = [0.00001, 0, 0, 0]

    # Growth rates:
    r1 = 10
    r2 = 8
    
    # Competition kernels
    K_kernel_choices = np.array([6, 4, 0, 0])  # choices for [K11, K12, K21, K22]
    beta_pars = [75.721873, 52.211524, 2.403204, 8.016032]         # values for [beta11, beta12, beta21, beta22]

    # Phenotype switching kernels
    if numerical_example == '1A': 
        M1_pars = [0.25, 0.002, 0.5, 0.002, 0.50]
        M2_pars = [0.35, 0.002]
    elif numerical_example == '1B':      
        M1_pars = [0.25, 0.002, 0.5, 0.002, 0.35]
        M2_pars = [0.50, 0.002]


diffusion_parameter_list = [D1, alpha11, alpha12, D2, alpha21, alpha22]
M_pars = [M1_pars, M2_pars]

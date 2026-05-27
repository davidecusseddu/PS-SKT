from math import prod as list_prod
import os
import numpy as np
from numerical_examples import *


### PHENOTYPIC SWITCHING COEFFICIENTS ####
varepsilon = 0.01

varepsilon1 = varepsilon
varepsilon2 = varepsilon
theta1 = 1/varepsilon1
theta2 = 1/varepsilon2



# DEFINITION OF THE INITIAL CONDITION
max_perturbation = 5e-3
perturbation_period = 3



# SOLVER PARAMETERS  

## Spatial discretisation
### Spatial coordinates: x in 1D and (x,y) in 2D
### Phenotype coordinate: z 

Nx = 150        # number of discretisation points x
Ny = Nx         # number of discretisation points y (if spatial domain is 2D)
Nz = 99         # number of discretisation points z

if Nz % 2 == 0:
    Nz += 1     # Since we use Simpson for integration, we need odd Nz

Ns = [Nx, Ny, Nz]
Ns = [Nx, Nz]
dimensions = len(Ns) 
N = list_prod(Ns)  
Ns.append(N)    # We combine all discretisations into a single array 



## Temporal discretisation

dt = 0.005      # Temporal discretisation step
T = 1           # Final time


# LINEAR SYSTEM SOLVER TOLERANCE
solver_tol = 1e-8

# PICARD ITERATION
picard_max_iter = 100
picard_iter = 1
picard_tol = 1e-8
picard = True


# MESH REFINEMENT PARAMETERS
refs_t = 0.9 # refs_t <= 1, as refinment occours by dt -> dt*refs_t
refs_x = 10
refs_z = 10
max_adaptive_steps = 10



# EXPORT SOLUTION EVERY np_export_step STEPS
np_export_step = 100
output_folder = 'Results/output_numerical_example' + numerical_example + '_eps=' + str(varepsilon) + '/'


  
def export_pars(output_folder):    

    import builtins
    
    print('All results will be stored in the folder: ' + output_folder)

    # Create the folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, 'all_variables.txt')

    # Finds all variables
    variables_to_export = {
        k: v for k, v in globals().items()
        if not k.startswith('__') and not callable(v) and not isinstance(v, type(builtins))
    }

    # Export to file
    try:
        with open(output_file, 'w') as f:
            for k, v in variables_to_export.items():
                try:
                    f.write(f"{k} = {repr(v)}\n")
                except Exception as inner_e:
                    f.write(f"{k} = <Unprintable: {inner_e}>\n")
        print(f"Exported all variables to {output_file}")
    except Exception as outer_e:
        print(f"Failed to export variables: {outer_e}")

    return

    

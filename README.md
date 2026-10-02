# PS-SKT

This code solves the nonlinear integro-differential system modelling a phenotype-structured version of the SKT system presented in 

[CGL26] Cusseddu, Gambino, Lorenzi. On a phenotype-structured Shigesada–Kawasaki–Teramoto model: Turing instability and pattern selection under fast phenotype switching, 2026 
arXiv: https://arxiv.org/abs/2605.28976

To reproduce the examples presented in [CGL26], please use the file numerical_examples.py and change the value of the variable numerical_example: ('1A', '1B', '2A', '2B', '3A', '3B').

The values of the remaining parameters of the system, including epsilon, and of the parameters of the numerical solver, are defined in parameters.py

To start the solver, the script to be run is main.py, which calculates the phenotype densities (n_1, n_2) by solving, iteratively, first for n_1, then for n_2. 

The two systems are assembled using the numerical method presented in [CGL26], coded by the functions present in functions.py. 

The solutions are then stored in the folder output_folder, which by default is created in parameters.py, and named 'Results/output_numerical_example' + numerical_example + '_eps=' + str(varepsilon) + '/'



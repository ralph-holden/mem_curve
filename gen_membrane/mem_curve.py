# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 12:35:32 2025

@author: ralph

Script containing functions for simulation of stable membrane curvatures -- to be used for initialising membrane MD simulation
"""
# # # Imports # # #
# For simulatution
import numpy as np
import copy
# For visualisation
import matplotlib.pyplot as plt
import matplotlib.animation as animation



# # # Information storing classes # # #

class params:
    '''
    Set parameters used in simulation
    '''
    # Size of thermal fluctuations, base unit of simulation
    kbT = 1  
    
    # Box size
    l_x = 50        # box size, x-direction
    l_y = 50        # box size, y-direction
    
    # Fourier expansion
    exp_order = 3   # order of 2D Fourier expansion
    
    # Bending energies
    H_0     = 0.0   # Optimum mean curvature
    kappa_H = 20.0   # Bending modulus of mean curvature (kbT/l units)
    kappa_K = -20.0   # Bending modulus of Gaussian curvature (kbT/l units)
    
    # Size of Monte Carlo moves
    delta = 0.003    # Standard deviation of perturbation applied to Fourier coefficients
    
    # X, Y grid for calculations
    npts_x, npts_y = 100, 100
    x = np.linspace(-l_x/2, l_x/2, npts_x)
    y = np.linspace(-l_y/2, l_y/2, npts_y)
    X, Y = np.meshgrid(x, y)


def init_model_membrane():
    '''
    Make class that stores Fourier cofficients describing surface (for 3rd order 2D expansion)
    ''' 
    membrane_dict = {
    'alpha' : np.zeros((params.exp_order, params.exp_order)),
    'beta'  : np.zeros((params.exp_order, params.exp_order)),
    'gamma' : np.zeros((params.exp_order, params.exp_order)),
    'zeta'  : np.zeros((params.exp_order, params.exp_order))
    }

    # Calculate first membrane energy
    # Calculate shape operator
    S = calc_shape_operator(membrane_dict, params.X, params.Y)
    # Calculate mean and Gaussian curvatures
    H   = calc_H(S)
    K_G = calc_K_G(S)
    # Calculate bending energy
    bending_energy = calc_Helfrich_energy(H, K_G)

    # Add energy to dictionary attributes
    membrane_dict['energy'] = bending_energy
    
    return membrane_dict



# # # Functions for calculating bending free energy # # #

def calc_height(membrane : dict, x : float, y : float):
    '''
    Calculate height (z-direction) of model membrane using 2D Fourier expansion
    
    INPUT
    membrane : dict,  holds Fourier coefficients and associated bending energy
    x        : float, x-position 
    y        : float, y-postiion
    
    OUPUT
    height   : float, height at point (x,y)
    '''
    sum1 = np.sum( membrane['alpha'][n,m]*np.cos(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum2 = np.sum( membrane['beta'][n,m] *np.cos(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum3 = np.sum( membrane['gamma'][n,m]*np.sin(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum4 = np.sum( membrane['alpha'][n,m]*np.sin(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    
    height = sum1 + sum2 + sum3 + sum4
    
    return height


def calc_fourier_derivatives(membrane : dict, x : float, y : float):
    '''
    Compute a 2D Fourier expansion h(x,y) and its derivatives up to second order.
    Parallelised loops over all derivatives.

    INPUT
    membrane : dict,  holds Fourier coefficients and associated bending energy
    x        : float, x_position
    y        : float, y-position 

    OUPUTS
    h_x      : float, first order partial derivative by x
    h_y      : float, first order partial derivative by y
    h_xx     : float, second order partial derivative by x
    h_xy     : float, second order partial derivative by x, y
    h_yy     : float, second order partial derivative by y
    '''
    # Initialise derivatives
    h_x   = np.zeros_like(x, dtype=float)
    h_y   = np.zeros_like(x, dtype=float)
    h_xx  = np.zeros_like(x, dtype=float)
    h_xy  = np.zeros_like(x, dtype=float)
    h_yy  = np.zeros_like(x, dtype=float)

    # Loop through Fourier expansion
    for n in range(params.exp_order):
        for m in range(params.exp_order):
            
            A = 2 * np.pi * n / params.l_x
            B = 2 * np.pi * m / params.l_y

            cosAx = np.cos(A * x)
            sinAx = np.sin(A * x)
            cosBy = np.cos(B * y)
            sinBy = np.sin(B * y)

            # First order derivatives
            h_x += (- membrane['alpha'][n, m] * A * sinAx * cosBy
                    - membrane['beta'][n, m]  * A * sinAx * sinBy
                    + membrane['gamma'][n, m] * A * cosAx * cosBy
                    + membrane['zeta'][n, m]  * A * cosAx * sinBy)

            h_y += (- membrane['alpha'][n, m] * B * cosAx * sinBy
                    + membrane['beta'][n, m]  * B * cosAx * cosBy
                    - membrane['gamma'][n, m] * B * sinAx * sinBy
                    + membrane['zeta'][n, m]  * B * sinAx * cosBy)

            # Second order derivatives
            h_xx += (- membrane['alpha'][n, m] * A**2 * cosAx * cosBy
                     - membrane['beta'][n, m]  * A**2 * cosAx * sinBy
                     - membrane['gamma'][n, m] * A**2 * sinAx * cosBy
                     - membrane['zeta'][n, m]  * A**2 * sinAx * sinBy)

            h_xy += (+ membrane['alpha'][n, m] * B**2 * cosAx * cosBy
                     - membrane['beta'][n, m]  * B**2 * cosAx * sinBy
                     - membrane['gamma'][n, m] * B**2 * sinAx * cosBy
                     + membrane['zeta'][n, m]  * B**2 * sinAx * sinBy)

            h_yy += (- membrane['alpha'][n, m] * A * B * sinAx * sinBy
                     - membrane['beta'][n, m]  * A * B * sinAx * cosBy
                     - membrane['gamma'][n, m] * A * B * cosAx * sinBy
                     - membrane['zeta'][n, m]  * A * B * cosAx * cosBy)

    return h_x, h_y, h_xx, h_xy, h_yy


def calc_shape_operator(membrane : dict, x : float, y : float):
    '''
    Calculate the shape operator at point (x,y)
    
    INPUT
    membrane : dict,  holds Fourier coefficients and associated bending energy
    x        : float, x-position
    y        : float, y-position
    
    OUPUT
    S_xy     : 2D array, shape operator at point x,y
    '''
    # Calculate partial derivatives of height
    h_x, h_y, h_xx, h_xy, h_yy = calc_fourier_derivatives(membrane, x, y)
    
    # Normalisation factor
    norm_factor = (1 + h_x**2 + h_y**2)**(-3/2)
    
    i0j0_term = (1+h_y**2)*h_xx - h_x*h_y*h_xy
    i0j1_term = (1+h_x**2)*h_xy - h_x*h_y*h_xx
    i1j0_term = (1+h_y**2)*h_xy - h_x*h_y*h_yy
    i1j1_term = (1+h_x**2)*h_yy - h_x*h_y*h_xy
    
    S_xy = norm_factor * np.array([ [i0j0_term, i0j1_term], [i1j0_term, i1j1_term] ])
    
    return S_xy


def calc_H(S_xy: np.ndarray):
    '''
    Calculate mean curvature, H at point (x,y)
    
    INPUT
    S_xy : 2D array, shape operator at point x,y
    
    OUPUT
    H    : float, mean curvature
    '''
    H = 0.5 * np.trace(S_xy)
    
    return H


def calc_K_G(S_xy: np.ndarray):
    '''
    Calculate Gaussian curvature, K_G at point (x,y)
    When S_xy is over all of grid X,Y -- must make determinant piecewise
    
    INPUT
    S_xy : 2D array, shape operator at point x,y
    
    OUPUT
    K_G  : float, Gaussian curvature
    '''
    #K_G = np.linalg.det(S_xy)
    a = S_xy[0, 0] 
    b = S_xy[0, 1]
    c = S_xy[1, 0]
    d = S_xy[1, 1]
    # Compute determinant
    K_G = a * d - b * c 
    
    return K_G


def calc_principle_curvatures(H : float, K_G : float):
    '''
    Calculate principle curvatures k_1, k_2 at point (x,y)
    
    INPUT
    H   : float, mean curvature
    K_G : float, Gaussian curvature
    
    OUPUT
    k_1 : float, principle curvature
    k_2 : float, principle curvature
    '''
    k_1 = H + np.sqrt( H**2 - K_G )
    k_2 = H - np.sqrt( H**2 - K_G )
    
    return k_1, k_2


def calc_Helfrich_energy(H : float, K_G : float):
    '''
    Calculate Helfrich bending energy of membrane
    Note: using bending potential energy, NOT lipid bilayer bending FREE energy
    
    *** Energy be for whole surface -> acceptance ratio dependant on box size (esp. for high frequencies)
        For small perturbations, can assume surface area = l_x * l_y (i.e. 2D area)
    
    INPUT
    H          : float, mean curvature
    K_G        : float, Gaussian curvature
    
    OUTPUT
    tot_energy : float, Helfrich bending energy over surface
    '''    
    energy_per_l = 2*params.kappa_H * ( H - params.H_0 )**2 + abs(params.kappa_K * K_G) # avoid negative bending energy??

    subgrid_area = params.l_x * params.l_y / (params.npts_x * params.npts_y) # for integration over total area
    
    tot_energy   = np.sum(energy_per_l) * subgrid_area
    
    return tot_energy



# # # Functions for Markov Chain Monte Carlo # # #

def montecarlomove(prev_membrane : dict):
    '''
    Make Markov Chain Monte Carlo (MCMC) move to Fourier coefficients describing surface
    
    INPUT
    prev_membrane : dict, contains curvature Fourier coefficients from previous step
    
    OUTPUT
    move_membrane : dict, new membrane with perturbed curvature Fourier coefficients AND associated energy
    '''
    # Fourier coefficient perturbations
    delta_alpha = np.random.normal(loc=0, scale=params.delta, size=(params.exp_order,params.exp_order))
    delta_beta  = np.random.normal(loc=0, scale=params.delta, size=(params.exp_order,params.exp_order))
    delta_gamma = np.random.normal(loc=0, scale=params.delta, size=(params.exp_order,params.exp_order))
    delta_zeta  = np.random.normal(loc=0, scale=params.delta, size=(params.exp_order,params.exp_order))

    # Apply to a model membrane copy
    move_membrane = copy.deepcopy(prev_membrane)
    move_membrane['alpha'] += delta_alpha
    move_membrane['beta']  += delta_beta
    move_membrane['gamma'] += delta_gamma
    move_membrane['zeta']  += delta_zeta

    # Calculate shape operator
    S = calc_shape_operator(move_membrane, params.X, params.Y)
    
    # Calculate mean and Gaussian curvatures
    H   = calc_H(S)
    K_G = calc_K_G(S)
    
    # Calculate bending energy
    move_energy = calc_Helfrich_energy(H, K_G)
    # Add to dictionary
    move_membrane['energy'] = move_energy
    
    return move_membrane
    

def montecarloeval(move_energy : float, prev_energy : float):
    '''
    Calculate MCMC move energy, evaluate move acceptance
    
    INPUT
    move_energy : float, energy associated with proposed membrane curvature
    prev_energy : float, energy associated with previous membrane curvature
    
    OUPUT
    accept_move : bool, Monte Carlo step outcome
    '''
    boltzmann_factor = np.exp( -(move_energy - prev_energy) )
    
    # choose random number
    rand_number = np.random.random()
    
    accept_move = rand_number < boltzmann_factor
    
    return accept_move


def montecarlostep(membrane_lst : list):
    '''
    Run MCMC step: propose move, evaluate, update Markov chain
    
    INPUT
    membrane_lst : list of dict, contains curvature Fourier coefficients and associated energy from previous steps
    
    OUPUT
    membrane_lst : list of Model_membrane, UPDATED ensemble of curvature Fourier coefficients and associated energy
    accept_move  : bool, Monte Carlo step outcome
    '''
    # Extract last state data
    prev_membrane = membrane_lst[-1]
    
    # Make Markov chain Monte Carlo move
    move_membrane = montecarlomove(prev_membrane)

    # Extract bending energies from dictionaries
    prev_energy = prev_membrane['energy']
    move_energy = move_membrane['energy']
    
    # Evaluate
    accept_move = montecarloeval(move_energy, prev_energy)
    
    # Update Markov chain
    membrane_lst += [move_membrane] if accept_move else [prev_membrane]

    return membrane_lst, accept_move



# # # Code for visualisation / data analysis # # #

def visualise(membrane_lst : list, nframes : int, save_dir=''):
    '''
    Visualise membrane curvature ensemble using heatmap plots
    
    INPUT
    membrane_lst : list of dict, contains curvature Fourier coefficients and associated energy
    nframes      : int, frequency/ interval size of frames
    save_dir     : str, name of directory to save animation, from view of working directory
    
    OUPUT
    anim         : matplotlib.animation.FuncAnimation, animation of heatmap plots of membrane height
    '''
    # X,Y grid
    npts_x, npts_y = 100, 100
    x = np.linspace(-params.l_x/2, params.l_x/2, npts_x)
    y = np.linspace(-params.l_y/2, params.l_y/2, npts_y)
    X, Y = np.meshgrid(x, y)
    
    # Calculate z-direction (heights)
    Z_dump = [calc_height(membrane, X, Y) for membrane in membrane_lst[::nframes]]

    # Find that with maximum range for colourbar
    ranges = [np.max(arr) - np.min(arr) for arr in Z_dump]
    max_range_idx = np.argmax(ranges)
    
    # Extract associated bending energies
    energy_lst = [membrane['energy'] for membrane in membrane_lst[::nframes]]
    
    # Animation plot
    fig, ax = plt.subplots(figsize=[8, 6])
    
    # Initial contour and colorbar
    contour = ax.contourf(X, Y, Z_dump[max_range_idx], levels=50, cmap='viridis', vmin=np.min(Z_dump), vmax=np.max(Z_dump))
    cbar = fig.colorbar(contour, ax=ax)

    # Labels
    cbar.set_label("Height")
    title = ax.set_title("Frame 0")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    def update(frame):
        nonlocal contour#, cbar
    
        # Remove previous contour 
        #for coll in contour.collections: -- for matplotlib <3.8
        #    coll.remove()                
        contour.remove()                 #-- for matplotlib >3.8
    
        # Draw new contour
        contour = ax.contourf(X, Y, Z_dump[frame], levels=50, cmap='viridis', vmin=np.min(Z_dump), vmax=np.max(Z_dump))
        # Add updated colorbar
        #cbar = fig.colorbar(contour, ax=ax)
        #cbar.set_label("Z value (scalar field)")
    
        # Update title
        title.set_text(f'Step: {frame*nframes} , Energy: {round(energy_lst[frame], 1)}')
    
        return contour.collections + [title]
        
    anim = animation.FuncAnimation(fig, update, frames=len(Z_dump), interval=100, blit=False)
    anim.save(f'./{save_dir}/contour_animation.gif', writer=animation.PillowWriter(fps=10))
    plt.show()
    
    return anim

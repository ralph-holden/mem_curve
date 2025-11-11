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
    exp_order = 4   # order of 2D Fourier expansion
    
    # Bending energies
    H_0     = 0.0   # Optimum mean curvature
    kappa_H = 20.0   # Bending rigidity of mean curvature (kbT*l units)
    kappa_K = -20.0   # Bending rigidity of Gaussian curvature (kbT*l^2 units)
    
    # Size of Monte Carlo moves
    delta = 0.01    # Standard deviation of perturbation applied to Fourier coefficients
    
    # X, Y grid for calculations
    npts = 5        # Number of points per l unit length -> npts^2 per l^2 unit area
    x = np.linspace(0, l_x, l_x * npts)
    y = np.linspace(0, l_y, l_y * npts)
    X, Y = np.meshgrid(x, y)


def init_model_membrane():
    '''
    Make class that stores Fourier cofficients describing surface (for 3rd order 2D expansion)

    INPUTS
    None

    OUTPUT
    membrane : dictionary, contains Fourier coefficients for flat surface and the associated bending energy
    ''' 
    membrane = {
    'alpha' : np.zeros((params.exp_order, params.exp_order)),
    'beta'  : np.zeros((params.exp_order, params.exp_order)),
    'gamma' : np.zeros((params.exp_order, params.exp_order)),
    'zeta'  : np.zeros((params.exp_order, params.exp_order))}

    # Calculate first membrane energy...
    # Calculate partial derivatives of height
    h_x, h_y, h_xx, h_xy, h_yy = calc_fourier_derivatives(membrane, params.X, params.Y)
    # Calculate shape operator
    S = calc_shape_operator(h_x, h_y, h_xx, h_xy, h_yy)
    # Calculate change in area from membrane bending
    dA = calc_area_element(h_x, h_y)
    # Calculate mean and Gaussian curvatures
    H   = calc_H(S)
    K_G = calc_K_G(S)
    # Calculate bending energy
    bending_energy = calc_Helfrich_energy(H, K_G, dA)

    # Add energy to dictionary attributes
    membrane['energy'] = bending_energy
    
    return membrane



# # # Functions for calculating bending free energy # # #

def calc_height(alpha : np.array, beta : np.array, gamma : np.array, zeta : np.array, X : np.array, Y : np.array, l_x : float, l_y : float):
    '''
    Calculate height (z-direction) of model membrane using 2D Fourier expansion
    
    INPUT
    alpha  : ndarray, Fourier series coefficient, params.exp_order square matrix
    beta   : ndarray, Fourier series coefficient, params.exp_order square matrix
    gamma  : ndarray, Fourier series coefficient, params.exp_order square matrix
    zeta   : ndarray, Fourier series coefficient, params.exp_order square matrix
    x      : ndarray, x-position 
    y      : ndarray, y-postiion
    l_x    : float, length of simulation box in X-axis
    l_y    : float, length of simulation box in Y-axis
    
    OUPUT
    height : float, height at point (x,y)
    '''
    # Sum integers
    n = np.arange(params.exp_order)
    m = np.arange(params.exp_order)
    
    # Add two extra dimensions for broadcasting with 2D X, Y
    cos_nx = np.cos(2*np.pi*n[:, None, None]*X/l_x) 
    cos_my = np.cos(2*np.pi*m[:, None, None]*Y/l_y)  
    sin_nx = np.sin(2*np.pi*n[:, None, None]*X/l_x)
    sin_my = np.sin(2*np.pi*m[:, None, None]*Y/l_y)
    
    # Compute all four sums using einsum
    suma = np.einsum('nm,nij,mij->ij', membrane['alpha'], cos_nx, cos_my)
    sumb = np.einsum('nm,nij,mij->ij', membrane['beta'],  cos_nx, sin_my)
    sumg = np.einsum('nm,nij,mij->ij', membrane['gamma'], sin_nx, cos_my)
    sumz = np.einsum('nm,nij,mij->ij', membrane['zeta'],  sin_nx, sin_my)
    
    height = suma + sumb + sumg + sumz

    return height
    

def calc_fourier_derivatives(alpha : np.array, beta : np.array, gamma : np.array, zeta : np.array, X : np.array, Y : np.array, l_x : float, l_y : float):
    '''
    Compute a 2D Fourier expansion h(x,y) and its derivatives up to second order.

    INPUT
    alpha  : ndarray, Fourier series coefficients, params.exp_order square matrix
    beta   : ndarray, Fourier series coefficients, params.exp_order square matrix
    gamma  : ndarray, Fourier series coefficients, params.exp_order square matrix
    zeta   : ndarray, Fourier series coefficients, params.exp_order square matrix
    x      : ndarray, 2D array of x_positions (meshgrid)
    y      : ndarray, 2D array of y-positions (meshgrid)
    l_x    : float, length of simulation box in X-axis
    l_y    : float, length of simulation box in Y-axis

    OUPUTS
    h_x    : float, first order partial derivative by x
    h_y    : float, first order partial derivative by y
    h_xx   : float, second order partial derivative by x, x
    h_xy   : float, second order partial derivative by x, y
    h_yy   : float, second order partial derivative by y, y
    '''
    # Sum integers
    n = np.arange(params.exp_order)
    m = np.arange(params.exp_order)
    
    # Compute differentiated trig arguments
    A = (2 * np.pi * n / l_x)[:, None, None]
    B = (2 * np.pi * m / l_y)[:, None, None]
    
    # Compute trig functions once
    cos_nx = np.cos(2*np.pi*n[:, None, None]*X/l_x)
    sin_nx = np.sin(2*np.pi*n[:, None, None]*X/l_x)
    cos_my = np.cos(2*np.pi*m[:, None, None]*Y/l_y)
    sin_my = np.sin(2*np.pi*m[:, None, None]*Y/l_y)
    
    # First order derivatives using einsum
    h_x = np.einsum('nm,nij,mij->ij', -membrane['alpha'] * (2*np.pi*n/l_x)[:, None], sin_nx, cos_my) + \
          np.einsum('nm,nij,mij->ij', -membrane['beta']  * (2*np.pi*n/l_x)[:, None], sin_nx, sin_my) + \
          np.einsum('nm,nij,mij->ij', +membrane['gamma'] * (2*np.pi*n/l_x)[:, None], cos_nx, cos_my) + \
          np.einsum('nm,nij,mij->ij', +membrane['zeta']  * (2*np.pi*n/l_x)[:, None], cos_nx, sin_my)
    
    h_y = np.einsum('nm,nij,mij->ij', -membrane['alpha'] * (2*np.pi*m/l_y)[None, :], cos_nx, sin_my) + \
          np.einsum('nm,nij,mij->ij', +membrane['beta']  * (2*np.pi*m/l_y)[None, :], cos_nx, cos_my) + \
          np.einsum('nm,nij,mij->ij', -membrane['gamma'] * (2*np.pi*m/l_y)[None, :], sin_nx, sin_my) + \
          np.einsum('nm,nij,mij->ij', +membrane['zeta']  * (2*np.pi*m/l_y)[None, :], sin_nx, cos_my)
    
    # Second order derivatives...
    # Compute differentiated trig arguments
    A_sq = (2*np.pi*n/l_x)**2
    B_sq = (2*np.pi*m/l_y)**2
    AB = (2*np.pi*n/l_x)[:, None] * (2*np.pi*m/l_y)[None, :]
    
    h_xx = np.einsum('nm,nij,mij->ij', -membrane['alpha'] * A_sq[:, None], cos_nx, cos_my) + \
           np.einsum('nm,nij,mij->ij', -membrane['beta']  * A_sq[:, None], cos_nx, sin_my) + \
           np.einsum('nm,nij,mij->ij', -membrane['gamma'] * A_sq[:, None], sin_nx, cos_my) + \
           np.einsum('nm,nij,mij->ij', -membrane['zeta']  * A_sq[:, None], sin_nx, sin_my)
    
    h_xy = np.einsum('nm,nij,mij->ij', +membrane['alpha'] * B_sq[None, :], cos_nx, cos_my) + \
           np.einsum('nm,nij,mij->ij', -membrane['beta']  * B_sq[None, :], cos_nx, sin_my) + \
           np.einsum('nm,nij,mij->ij', -membrane['gamma'] * B_sq[None, :], sin_nx, cos_my) + \
           np.einsum('nm,nij,mij->ij', +membrane['zeta']  * B_sq[None, :], sin_nx, sin_my)
    
    h_yy = np.einsum('nm,nij,mij->ij', -membrane['alpha'] * AB, sin_nx, sin_my) + \
           np.einsum('nm,nij,mij->ij', -membrane['beta']  * AB, sin_nx, cos_my) + \
           np.einsum('nm,nij,mij->ij', -membrane['gamma'] * AB, cos_nx, sin_my) + \
           np.einsum('nm,nij,mij->ij', -membrane['zeta']  * AB, cos_nx, cos_my)

    return h_x, h_y, h_xx, h_xy, h_yy 


def calc_shape_operator(h_x : float, h_y : float, h_xx : float, h_xy : float, h_yy : float):
    '''
    Calculate the shape operator at point (x,y)
    
    INPUT
    h_x  : float, first order partial derivative by x
    h_y  : float, first order partial derivative by y
    h_xx : float, second order partial derivative by x, x
    h_xy : float, second order partial derivative by x, y
    h_yy : float, second order partial derivative by y, y
    
    OUPUT
    S_xy : 2D array, shape operator at point x,y
    '''
    # Normalisation factor
    norm_factor = (1 + h_x**2 + h_y**2)**(-3/2)
    
    i0j0_term = (1+h_y**2)*h_xx - h_x*h_y*h_xy
    i0j1_term = (1+h_x**2)*h_xy - h_x*h_y*h_xx
    i1j0_term = (1+h_y**2)*h_xy - h_x*h_y*h_yy
    i1j1_term = (1+h_x**2)*h_yy - h_x*h_y*h_xy

    S_xy = norm_factor * np.array([ [i0j0_term, i0j1_term], [i1j0_term, i1j1_term] ])
    
    return S_xy


def calc_area_element(h_x : float, h_y : float):
    '''
    Calculate the area element (dA) of a "monge patch" at x, y
    Note: x, y positions specified when calculating h_x, h_y in calc_shape_operator 

    INPUTS
    h_x : float, partial derivative of height by x at point x,y
    h_y : float, partial derivative of height by y at point x,y

    OUPUTS
    dA  : float, area element of "monge patch"
    '''
    dA = np.sqrt(1 + h_x**2 + h_y**2)
    #dA = 1 + np.sum(dA_parts)

    return dA


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


def calc_Helfrich_energy(H : float, K_G : float, dA = 1):
    '''
    Calculate Helfrich bending energy of membrane
    Note: using bending potential energy, NOT lipid bilayer bending FREE energy
    Using change in 2D area from bending, dA, to rescale the 2D area at every point
    
    *** Energy be for whole surface -> acceptance ratio dependant on box size (esp. for high frequencies)
    
    INPUT
    H          : float, mean curvature
    K_G        : float, Gaussian curvature
    dA         : float, change in 2D area from membrane bending
    
    OUTPUT
    tot_energy : float, Helfrich bending energy over surface
    '''    
    energy_per_l = 2*params.kappa_H * ( H - params.H_0 )**2 + abs(params.kappa_K * K_G) # abs to avoid negative bending energy

    subgrid_area = 1 / (dA * params.npts**2) # for integration over total (2D) area
    
    tot_energy   = np.sum(energy_per_l * subgrid_area)
    
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

    # Calculate local height derivatives from Fourier surface
    h_x, h_y, h_xx, h_xy, h_yy = calc_fourier_derivatives(move_membrane, params.X, params.Y)
    
    # Calculate shape operator
    S = calc_shape_operator(h_x, h_y, h_xx, h_xy, h_yy)

    # Calculate change in area from membrane bending
    dA = calc_area_element(h_x, h_y)
    
    # Calculate mean and Gaussian curvatures
    H   = calc_H(S)
    K_G = calc_K_G(S)
    
    # Calculate bending energy
    move_energy = calc_Helfrich_energy(H, K_G, dA) # set dA=1 to ignore
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
    Note: un/comment code for use between matplotlib versions < & > 3.8 
    
    INPUT
    membrane_lst : list of dict, contains curvature Fourier coefficients and associated energy
    nframes      : int, frequency/ interval size of frames
    save_dir     : str, name of directory to save animation, from view of working directory
    
    OUPUT
    anim         : matplotlib.animation.FuncAnimation, animation of heatmap plots of membrane height
    '''
    # X,Y grid
    npts = 4 
    x = np.linspace(0, params.l_x, params.l_x*npts)
    y = np.linspace(0, params.l_y, params.l_y*npts)
    X, Y = np.meshgrid(x, y)
    
    # Calculate z-direction (heights)
    Z_dump = [calc_height(membrane, X, Y) for membrane in membrane_lst[::nframes]]
    
    # Extract associated bending energies
    energy_lst = [membrane['energy'] for membrane in membrane_lst[::nframes]]

    # Find that with maximum range for colourbar
    #ranges = [np.max(arr) - np.min(arr) for arr in Z_dump]
    #max_range_idx = np.argmax(ranges)
    #maxval = np.max(Z_dump[max_range_idx]) if np.max(Z_dump[max_range_idx])>abs(np.min(Z_dump[max_range_idx])) else abs(np.min(Z_dump[max_range_idx]))
    Z_min, Z_max = np.min(Z_dump), np.max(Z_dump)
    maxval = Z_max if Z_max>abs(Z_min) else abs(Z_min)
    Z_for_colourbar = Z_dump[0] + 0.0 # trick to make copy
    Z_for_colourbar[0], Z_for_colourbar[1] = Z_min, Z_max # edit values so they show up on colourbar
    
    # Animation plot
    fig, ax = plt.subplots(figsize=[8, 6])
    
    # Initial contour and colorbar
    #contour = ax.contourf(X, Y, Z_dump[max_range_idx], levels=50, cmap='viridis', vmin=-maxval, vmax=maxval)
    contour = ax.contourf(X, Y, Z_for_colourbar, levels=50, cmap='viridis', vmin=-maxval, vmax=maxval)
    cbar = fig.colorbar(contour, ax=ax)

    # Labels
    cbar.set_label("Height")
    title = ax.set_title("Frame 0")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    def update(frame):
        nonlocal contour 
        
        # Remove previous contour
        #for coll in contour.collections: -- for matplotlib <3.8
        #    coll.remove() 
        if contour is not None:
            contour.remove()  # -- for matplotlib >= 3.8
    
        # Draw new contour
        contour = ax.contourf(
            X, Y, Z_dump[frame],
            levels=50, cmap='viridis',
            vmin=-abs(maxval), vmax=+abs(maxval))
    
        # Update title
        title.set_text(f"Step: {frame * nframes} , Energy: {round(energy_lst[frame], 1)}")

        #return contour.collections + [title] # -- for matplotlib <3.8
        return [contour, title] 
        
    anim = animation.FuncAnimation(fig, update, frames=len(Z_dump), interval=100, blit=False)
    anim.save(f'./{save_dir}/contour_animation.gif', writer=animation.PillowWriter(fps=10))
    plt.show()
    
    return anim
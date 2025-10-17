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
    l_x = 10        # box size, x-direction
    l_y = 10        # box size, y-direction
    
    # Fourier expansion
    exp_order = 3   # order of 2D Fourier expansion
    
    # Bending energies
    H_0     = 0.0   # Optimum mean curvature
    kappa_H = 1.0   # Bending modulus of mean curvature (kbT units)
    kappa_K = 1.0   # Bending modulus of Gaussian curvature (kbT units)
    
    # Size of Monte Carlo moves
    delta = 0.01    # Mean size of perturbation applied to Fourier coefficients
    
    # X, Y grid for calculations
    npts = 100
    x = np.linspace(0, l_x, npts)
    y = np.linspace(0, l_y, npts)
    X, Y = np.meshgrid(x, y)


class Model_membrane:
    '''
    Store Fourier cofficients describing surface (for 3rd order 2D expansion)
    ''' 
    def __init__(self):
        
        self.alpha = np.zeros((params.exp_order, params.exp_order))
        self.beta  = np.zeros((params.exp_order, params.exp_order))
        self.gamma = np.zeros((params.exp_order, params.exp_order))
        self.zeta  = np.zeros((params.exp_order, params.exp_order))
        
       
        
# # # Functions for calculating bending free energy # # #

def calc_height(membrane : Model_membrane, x : float, y : float):
    '''
    Calculate height (z-direction) of model membrane using 2D Fourier expansion
    
    INPUT
    membrane : Model_membrane, holds Fourier coefficients
    x        : float, x-position 
    y        : float, y-postiion
    
    OUPUT
    height   : float, height at point (x,y)
    '''
    
    sum1 = np.sum( membrane.alpha[n,m]*np.cos(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum2 = np.sum( membrane.beta[n,m] *np.cos(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum3 = np.sum( membrane.gamma[n,m]*np.sin(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum4 = np.sum( membrane.alpha[n,m]*np.sin(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    
    height = sum1 + sum2 + sum3 + sum4
    
    return height


def calc_h_x(membrane : Model_membrane, x : float, y : float):
    '''
    Calculate local first order partial differential Fourier expansion of height by x
    
    INPUT
    membrane : Model_membrane, holds Fourier coefficients
    x        : float, x_position
    y        : float, y-position 
    
    OUTPUT
    h_x      : float, local partial derivative by x
    '''
    sum1 = np.sum( -membrane.alpha[n,m]*(2*np.pi*n/params.l_x)*np.sin(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum2 = np.sum( -membrane.beta[n,m] *(2*np.pi*n/params.l_x)*np.sin(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum3 = np.sum(  membrane.gamma[n,m]*(2*np.pi*n/params.l_x)*np.cos(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum4 = np.sum(  membrane.zeta[n,m]*(2*np.pi*n/params.l_x)*np.cos(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    
    h_x = sum1 + sum2 + sum3 + sum4
    
    return h_x


def calc_h_y(membrane : Model_membrane, x : float, y : float):
    '''
    Calculate local first order partial differential Fourier expansion of height by y
    
    INPUT
    membrane : Model_membrane, holds Fourier coefficients
    x        : float, x_position
    y        : float, y-position 
    
    OUTPUT
    h_y      : float, local partial derivative by y
    '''
    sum1 = np.sum( -membrane.alpha[n,m]*(2*np.pi*m/params.l_y)*np.cos(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum2 = np.sum(  membrane.beta[n,m] *(2*np.pi*m/params.l_y)*np.cos(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum3 = np.sum( -membrane.gamma[n,m]*(2*np.pi*m/params.l_y)*np.sin(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum4 = np.sum(  membrane.zeta[n,m]*(2*np.pi*m/params.l_y)*np.sin(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    
    h_y = sum1 + sum2 + sum3 + sum4
    
    return h_y


def calc_h_xx(membrane : Model_membrane, x : float, y : float):
    '''
    Calculate local second order partial differential Fourier expansion of height by x
    
    INPUT
    membrane : Model_membrane, holds Fourier coefficients
    x        : float, x_position
    y        : float, y-position 
    
    OUTPUT
    h_xx     : float, local second order partial derivative by x
    '''
    sum1 = np.sum( -membrane.alpha[n,m]*(2*np.pi*n/params.l_x)**2*np.cos(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum2 = np.sum( -membrane.beta[n,m] *(2*np.pi*n/params.l_x)**2*np.cos(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum3 = np.sum( -membrane.gamma[n,m]*(2*np.pi*n/params.l_x)**2*np.sin(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum4 = np.sum( -membrane.zeta[n,m]*(2*np.pi*n/params.l_x)**2*np.sin(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    
    h_xx = sum1 + sum2 + sum3 + sum4
    
    return h_xx


def calc_h_yy(membrane : Model_membrane, x : float, y : float):
    '''
    Calculate local second order partial differential Fourier expansion of height by y
    
    INPUT
    membrane : Model_membrane, holds Fourier coefficients
    x        : float, x_position
    y        : float, y-position 
    
    OUTPUT
    h_yy     : float, local second order partial derivative by y
    '''
    sum1 = np.sum( -membrane.alpha[n,m]*(2*np.pi*m/params.l_y)**2*np.cos(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum2 = np.sum( -membrane.beta[n,m] *(2*np.pi*m/params.l_y)**2*np.cos(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum3 = np.sum( -membrane.gamma[n,m]*(2*np.pi*m/params.l_y)**2*np.sin(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum4 = np.sum( -membrane.zeta[n,m]*(2*np.pi*m/params.l_y)**2*np.sin(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    
    h_yy = sum1 + sum2 + sum3 + sum4
    
    return h_yy


def calc_h_xy(membrane : Model_membrane, x : float, y : float):
    '''
    Calculate local second order partial differential Fourier expansion of height by x, y
    
    INPUT
    membrane : Model_membrane, holds Fourier coefficients
    x        : float, x_position
    y        : float, y-position 
    
    OUTPUT
    h_xy     : float, local second order partial derivative by x
    '''
    sum1 = np.sum(  membrane.alpha[n,m]*(2*np.pi*n/params.l_x)*(2*np.pi*m/params.l_y)*np.sin(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum2 = np.sum( -membrane.beta[n,m] *(2*np.pi*n/params.l_x)*(2*np.pi*m/params.l_y)*np.sin(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum3 = np.sum( -membrane.gamma[n,m]*(2*np.pi*n/params.l_x)*(2*np.pi*m/params.l_y)*np.cos(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum4 = np.sum(  membrane.zeta[n,m]*(2*np.pi*n/params.l_x)*(2*np.pi*m/params.l_y)*np.cos(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    
    h_xy = sum1 + sum2 + sum3 + sum4
    
    return h_xy


def calc_shape_operator(membrane : Model_membrane, x : float, y : float):
    '''
    Calculate the shape operator at point (x,y)
    
    INPUT
    membrane : Model_membrane, holds Fourier coefficients
    x        : float, x-position
    y        : float, y-position
    
    OUPUT
    S_xy     : 2D array, shape operator at point x,y
    '''
    # Calculate first order partial derivatives of height
    h_x  = calc_h_x(membrane, x, y)
    h_y  = calc_h_y(membrane, x, y)
    
    # Calculate second order partial derivatives of height
    h_xx = calc_h_xx(membrane, x, y)
    h_xy = calc_h_xy(membrane, x, y)
    h_yy = calc_h_yy(membrane, x, y)
    
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
    
    INPUT
    S_xy : 2D array, shape operator at point x,y
    
    OUPUT
    K_G  : float, Gaussian curvature
    '''
    K_G = np.linalg.det(S_xy)
    
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
    
    *** Check equation is correct
    *** Find values for kappa_H, H_0, kappa_K
    *** Energy be for whole surface -> acceptance ratio dependant on box size (esp. for high frequencies)
        For small perturbations, can assume surface area = l_x * l_y (i.e. 2D area)
    
    INPUT
    H          : float, mean curvature
    K_G        : float, Gaussian curvature
    
    OUTPUT
    tot_energy : float, Helfrich bending energy over surface
    '''    
    energy_per_l = 2*params.kappa_H * ( H - params.H_0 )**2 + params.kappa_K * K_G

    tot_energy = energy_per_l * params.l_x * params.l_y
    
    return tot_energy



# # # Functions for Markov Chain Monte Carlo # # #

def montecarlomove(prev_membrane : Model_membrane):
    '''
    Make Markov Chain Monte Carlo (MCMC) move to Fourier coefficients describing surface
    
    INPUT
    prev_membrane : Model_membrane, contains curvature Fourier coefficients from previous step
    
    OUTPUT
    move_membrane : Model_membrane, new membrane with perturbed curvature Fourier coefficients
    move_energy   : float, bending energy associated with move_membrane
    '''
    # Fourier coefficient perturbations
    delta_alpha = np.random.normal(loc=0, scale=params.delta, size=params.exp_order)
    delta_beta  = np.random.normal(loc=0, scale=params.delta, size=params.exp_order)
    delta_gamma = np.random.normal(loc=0, scale=params.delta, size=params.exp_order)
    delta_zeta  = np.random.normal(loc=0, scale=params.delta, size=params.exp_order)

    # Apply to a model membrane copy
    move_membrane = copy.deepcopy(prev_membrane)
    move_membrane.alpha += delta_alpha
    move_membrane.beta  += delta_beta
    move_membrane.gamma += delta_gamma
    move_membrane.zeta  += delta_zeta

    # Calculate shape operator
    S = calc_shape_operator(move_membrane, params.X, params.Y)
    
    # Calculate mean and Gaussian curvatures
    H   = calc_H(S)
    K_G = calc_K_G(S)
    
    # Calculate bending energy
    move_energy = calc_Helfrich_energy(H, K_G)
    
    return move_membrane, move_energy
    

def montecarloeval(move_energy : float, prev_energy : float):
    '''
    Calculate MCMC move energy, evaluate move acceptance
    
    INPUT
    move_energy : flpat, energy associated with proposed membrane curvature
    prev_energy : float, energy associated with previous membrane curvature
    
    OUPUT
    accept_move : bool, Monte Carlo step outcome
    '''
    boltzmann_factor = np.exp( -(move_energy - prev_energy) )
    
    # choose random number
    rand_number = np.random.random()
    
    accept_move = boltzmann_factor < rand_number
    
    return accept_move


def montecarlostep(membrane_lst : list, energy_lst : list):
    '''
    Run MCMC step: propose move, evaluate, update Markov chain
    
    INPUT
    membrane_lst : list of Model_membrane, contains curvature Fourier coefficients from previous steps
    energy_lst   : list of float, energy associated with previous membrane curvatures
    
    OUPUT
    membrane_lst : list of Model_membrane, UPDATED ensemble of curvature Fourier coefficients
    energy_st    : list of float, UPDATED ensemble of energies
    '''
    # Extract last state data
    prev_membrane, prev_energy = membrane_lst[-1], energy_lst[-1]
    
    # Make Markov chain Monte Carlo move
    move_membrane, move_energy = montecarlomove(prev_membrane)
    
    # Evaluate
    accept_move = montecarloeval(move_energy, prev_energy)
    
    # Update Markov chain
    membrane_lst += [move_membrane] if accept_move else [prev_membrane]
    energy_lst   += [move_energy]   if accept_move else [prev_energy]

    return membrane_lst, energy_lst



# # # Code for visualisation / data analysis # # #

def visualise(membrane_lst : list, nframes : int):
    '''
    Visualise membrane curvature ensemble using heatmap plots
    
    INPUT
    membrane_lst : list of Model_membrane, contains curvature Fourier coefficients from previous steps
    nframes      : int, frequency/ interval size of frames
    
    OUPUT
    None
    '''
    # Dump for membrane height
    Z_dump = []
    
    # X,Y grid
    npts = 100
    x = np.linspace(0, params.l_x, npts)
    y = np.linspace(0, params.l_y, npts)
    X, Y = np.meshgrid(x, y)
    
    # Calculate z-direction (heights) using membrane Fourier coefficients, every nframes
    for membrane in membrane_lst[::nframes]:
    
        Z_dump += [calc_height(membrane, X, Y)]
        
    # Make contour plot movie
    # Set up figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initial contour plot
    contour = ax.contourf(X, Y, Z_dump[0], levels=50, cmap='viridis')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Z value (scalar field)")
    
    # Function to update contour only (not clearing the axis)
    def update(frame):
        for coll in contour.collections:
            coll.remove()
        new_contour = ax.contourf(X, Y, Z_dump[frame], levels=50, cmap='viridis')
        contour.collections[:] = new_contour.collections
        ax.set_title(f"Frame {frame}")
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(Z_dump), interval=100)
    
    # Save to video
    #ani.save("contour_with_colorbar.mp4", writer='ffmpeg', dpi=200) # *** Cannot save animation!
    plt.close()
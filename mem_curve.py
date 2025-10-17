# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 12:35:32 2025

@author: ralph

Script containing functions for simulation of stable membrane curvatures -- to be used for initialising membrane MD simulation
"""
# # # Imports # # #

import numpy as np
import matplotlib.pylot as plt



# # # Information storing classes # # #

class variables:
    '''
    Store variables used in simulation
    '''
    kbT  = 1  
    l_x  = 10 # box size, x-direction
    l_y  = 10 # box size, y-direction


class Model_membrane:
    '''
    Store Fourier cofficients describing surface (for 3rd order 2D expansion)
    ''' 
    def __init__(self):
        
        self.alpha = np.array([ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ])
        self.beta  = np.array([ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ])
        self.gamma = np.array([ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ])
        self.zeta  = np.array([ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ])
        
       
        
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
    exp_order = 3 # order of 2D Fourier expansion
    
    sum1 = np.sum( membrane.alpha[n,m]*np.cos(2*np.pi*n*x/variables.l_x)*np.cos(2*np.pi*m*y/variables.l_y) for n in range(exp_order) for m in range(exp_order) )
    sum2 = np.sum( membrane.beta[n,m] *np.cos(2*np.pi*n*x/variables.l_x)*np.sin(2*np.pi*m*y/variables.l_y) for n in range(exp_order) for m in range(exp_order) )
    sum3 = np.sum( membrane.gamma[n,m]*np.sin(2*np.pi*n*x/variables.l_x)*np.cos(2*np.pi*m*y/variables.l_y) for n in range(exp_order) for m in range(exp_order) )
    sum4 = np.sum( membrane.alpha[n,m]*np.sin(2*np.pi*n*x/variables.l_x)*np.sin(2*np.pi*m*y/variables.l_y) for n in range(exp_order) for m in range(exp_order) )
    
    height = sum1 + sum2 + sum3 + sum4
    
    return height


def calc_h_x(membrane : Model_membrane, x : float):
    '''
    Calculate local first order partial differential of height by x
    
    INPUT
    membrane : Model_membrane, holds Fourier coefficients
    x        : float, x-position 
    '''
    return 0


def calc_h_y(membrane : Model_membrane, y : float):
    '''
    Calculate local first order partial differential of height by y
    
    INPUT
    membrane : Model_membrane, holds Fourier coefficients
    y        : float, x-position 
    '''
    return 0


def calc_h_xx(membrane : Model_membrane, x : float):
    '''
    Calculate local second order partial differential of height by x
    
    INPUT
    membrane : Model_membrane, holds Fourier coefficients
    x        : float, x-position
    '''
    return 0


def calc_h_xy(membrane : Model_membrane, x : float, y : float):
    '''
    Calculate local second order partial differential of height by x, y
    
    INPUT
    membrane : Model_membrane, holds Fourier coefficients
    x        : float, x-position
    y        : float, y-position
    '''
    return 0


def calc_h_yy(membrane : Model_membrane, y : float):
    '''
    Calculate local second order partial differential of height by y 
    
    INPUT
    membrane : Model_membrane, holds Fourier coefficients
    y        : float, y-position
    '''
    return 0


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
    h_x  = calc_h_x(membrane, x)
    h_y  = calc_h_y(membrane, y)
    
    # Calculate second order partial derivatives of height
    h_xx = calc_h_xx(membrane, x)
    h_xy = calc_h_xy(membrane, x, y)
    h_yy = calc_h_yy(membrane, y)
    
    # Normalisation factor
    norm_factor = (1 + h_x**2 + h_y**2)**(-3/2)
    
    i0j0_term = (1+h_y**2)*h_xx - h_x*h_y*h_xy
    i0j1_term = (1+h_x**2)*h_xy - h_x*h_y*h_xx
    i1j0_term = (1+h_y**2)*h_xy - h_x*h_y*h_yy
    i1j1_term = (1+h_x**2)*h_yy - h_x*h_y*h_xy
    
    S_xy = norm_factor * np.array( [i0j0_term, i0j1_term], [i1j0_term, i1j1_term] )
    
    return S_xy


def calc_H(S_xy: np.ndarray):
    '''
    Calculate mean curvature, H at point (x,y)
    
    INPUT
    S_xy : 2D array, shape operator at point x,y
    
    OUPUT
    H    : float, mean curvature
    '''
    H = 0.5 * np.linalg.trace(S_xy)
    
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
    
    INPUT
    H      : float, mean curvature
    K_G    : float, Gaussian curvature
    
    OUTPUT
    energy : float, Helfrich bending energy
    '''    
    return 0



# # # Functions for Markov Chain Monte Carlo # # #

def montecarlomove():
    '''
    Make Markov Chain Monte Carlo (MCMC) move to Fourier coefficients describing surface
    '''
    pass

def montecarloeval():
    '''
    Calculate MCMC move energy, evaluate move acceptance
    '''
    pass

def montecarlostep():
    '''
    Run MCMC step: propose move, evaluate, add to Markov chain
    Note: only need to store (alpha, beta, gamma, zeta) values -- can construct all other values using
    '''
    pass



# # # Functions for running simulation, replica exchange, annealing # # #

# Simulation:
#   Initialise model membrane, run Monte Carlo for <n> steps
#   Every <n'> steps, save values for data analysis: energy, mean curvature, Gaussian curvature

# Replica exchange:
#   Access surfaces otherwise separated by potential barriers...
#   Tune kbT factor up & down to alter acceptance ratio from equilibrium ensemble
#   Run parallel simulation windows, can swap replicas between kbT environments

# Annealing:
#   Slowly return kbT factor to equilibrium
#   If too quick, will form "glass" -- initially all moves "downhill" & become trapped in metastable state space



# # # Data Analysis # # #

# Plot bending energy vs Monte Carlo step
# Plot mean curvature vs Monte Carlo step
# Plot Gaussian curvature vs Monte Carlo step
# ^ are these values expected? Are states stable &/ in equilibrium ensemble?

# Take mean of height (from equilibrium sampled region)
#   Is average membrane structure flat?

# Take mean squared height
#   How does membrane thickness compare to CWT?

# Visualising membrane curvature
# For every <n''> steps, use Fourier coefficients to...
# Construct contour / heatmap plot -- x,y axis w/ z for membrane height
# Save frames to movie
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:31:50 2025

@author: ralph
"""

def calc_h_x(membrane : dict, x : float, y : float):
    '''
    Calculate local first order partial differential Fourier expansion of height by x
    
    INPUT
    membrane : dict,  holds Fourier coefficients and associated bending energy
    x        : float, x_position
    y        : float, y-position 
    
    OUTPUT
    h_x      : float, local partial derivative by x
    '''
    sum1 = np.sum( -membrane['alpha'][n,m]*(2*np.pi*n/params.l_x)*np.sin(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum2 = np.sum( -membrane['beta'][n,m] *(2*np.pi*n/params.l_x)*np.sin(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum3 = np.sum(  membrane['gamma'][n,m]*(2*np.pi*n/params.l_x)*np.cos(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum4 = np.sum(  membrane['zeta'][n,m]*(2*np.pi*n/params.l_x)*np.cos(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    
    h_x = sum1 + sum2 + sum3 + sum4
    
    return h_x


def calc_h_y(membrane : dict, x : float, y : float):
    '''
    Calculate local first order partial differential Fourier expansion of height by y
    
    INPUT
    membrane : dict,  holds Fourier coefficients and associated bending energy
    x        : float, x_position
    y        : float, y-position 
    
    OUTPUT
    h_y      : float, local partial derivative by y
    '''
    sum1 = np.sum( -membrane['alpha'][n,m]*(2*np.pi*m/params.l_y)*np.cos(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum2 = np.sum(  membrane['beta'][n,m] *(2*np.pi*m/params.l_y)*np.cos(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum3 = np.sum( -membrane['gamma'][n,m]*(2*np.pi*m/params.l_y)*np.sin(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum4 = np.sum(  membrane['zeta'][n,m]*(2*np.pi*m/params.l_y)*np.sin(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    
    h_y = sum1 + sum2 + sum3 + sum4
    
    return h_y


def calc_h_xx(membrane : dict, x : float, y : float):
    '''
    Calculate local second order partial differential Fourier expansion of height by x
    
    INPUT
    membrane : dict,  holds Fourier coefficients and associated bending energy
    x        : float, x_position
    y        : float, y-position 
    
    OUTPUT
    h_xx     : float, local second order partial derivative by x
    '''
    sum1 = np.sum( -membrane['alpha'][n,m]*(2*np.pi*n/params.l_x)**2*np.cos(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum2 = np.sum( -membrane['beta'][n,m] *(2*np.pi*n/params.l_x)**2*np.cos(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum3 = np.sum( -membrane['gamma'][n,m]*(2*np.pi*n/params.l_x)**2*np.sin(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum4 = np.sum( -membrane['zeta'][n,m]*(2*np.pi*n/params.l_x)**2*np.sin(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    
    h_xx = sum1 + sum2 + sum3 + sum4
    
    return h_xx


def calc_h_yy(membrane : dict, x : float, y : float):
    '''
    Calculate local second order partial differential Fourier expansion of height by y
    
    INPUT
    membrane : dict,  holds Fourier coefficients and associated bending energy
    x        : float, x_position
    y        : float, y-position 
    
    OUTPUT
    h_yy     : float, local second order partial derivative by y
    '''
    sum1 = np.sum( -membrane['alpha'][n,m]*(2*np.pi*m/params.l_y)**2*np.cos(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum2 = np.sum( -membrane['beta'][n,m] *(2*np.pi*m/params.l_y)**2*np.cos(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum3 = np.sum( -membrane['gamma'][n,m]*(2*np.pi*m/params.l_y)**2*np.sin(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum4 = np.sum( -membrane['zeta'][n,m]*(2*np.pi*m/params.l_y)**2*np.sin(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    
    h_yy = sum1 + sum2 + sum3 + sum4
    
    return h_yy


def calc_h_xy(membrane : dict, x : float, y : float):
    '''
    Calculate local second order partial differential Fourier expansion of height by x, y
    
    INPUT
    membrane : dict,  holds Fourier coefficients and associated bending energy
    x        : float, x_position
    y        : float, y-position 
    
    OUTPUT
    h_xy     : float, local second order partial derivative by x
    '''
    sum1 = np.sum(  membrane['alpha'][n,m]*(2*np.pi*n/params.l_x)*(2*np.pi*m/params.l_y)*np.sin(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum2 = np.sum( -membrane['beta'][n,m] *(2*np.pi*n/params.l_x)*(2*np.pi*m/params.l_y)*np.sin(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum3 = np.sum( -membrane['gamma'][n,m]*(2*np.pi*n/params.l_x)*(2*np.pi*m/params.l_y)*np.cos(2*np.pi*n*x/params.l_x)*np.sin(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    sum4 = np.sum(  membrane['zeta'][n,m]*(2*np.pi*n/params.l_x)*(2*np.pi*m/params.l_y)*np.cos(2*np.pi*n*x/params.l_x)*np.cos(2*np.pi*m*y/params.l_y) for n in range(params.exp_order) for m in range(params.exp_order) )
    
    h_xy = sum1 + sum2 + sum3 + sum4
    
    return h_xy


def calc_shape_operator_piecewise(membrane : dict, x : float, y : float):
    '''
    Calculate the shape operator at point (x,y)
    Uses derivative calculations that are NOT parallelised -- slower than calc_shape_operator
    
    INPUT
    membrane : dict,  holds Fourier coefficients and associated bending energy
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
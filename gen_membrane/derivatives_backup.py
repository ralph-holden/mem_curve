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


def calc_height_slow(alpha : float, beta : float, gamma : float, zeta : float, x : float, y : float, l_x : float, l_y : float):
    '''
    Calculate height (z-direction) of model membrane using 2D Fourier expansion
    
    INPUT
    alpha  : ndarray, Fourier series coefficient, params.exp_order square matrix
    beta   : ndarray, Fourier series coefficient, params.exp_order square matrix
    gamma  : ndarray, Fourier series coefficient, params.exp_order square matrix
    zeta   : ndarray, Fourier series coefficient, params.exp_order square matrix
    x      : float, x-position 
    y      : float, y-postiion
    l_x    : float, length of simulation box in X-axis
    l_y    : float, length of simulation box in Y-axis
    
    OUPUT
    height : float, height at point (x,y)
    '''
    suma, sumb, sumg, sumz = 0, 0, 0, 0
    for n in range(params.exp_order):
        for m in range(params.exp_order):
            suma += alpha[n,m] * np.cos(2*np.pi*n*x/l_x)*np.cos(2*np.pi*m*y/l_y)
            sumb += beta[n,m]  * np.cos(2*np.pi*n*x/l_x)*np.sin(2*np.pi*m*y/l_y) 
            sumg += gamma[n,m] * np.sin(2*np.pi*n*x/l_x)*np.cos(2*np.pi*m*y/l_y)
            sumz += zeta[n,m]  * np.sin(2*np.pi*n*x/l_x)*np.sin(2*np.pi*m*y/l_y)
    
    height = suma + sumb + sumg + sumz
    
    return height


def calc_fourier_derivatives_slow(alpha : float, beta : float, gamma : float, zeta : float, x : float, y : float, l_x : float, l_y : float):
    '''
    Compute a 2D Fourier expansion h(x,y) and its derivatives up to second order.
    Parallelised loops over all derivatives.

    INPUT
    alpha  : ndarray, Fourier series coefficient, params.exp_order square matrix
    beta   : ndarray, Fourier series coefficient, params.exp_order square matrix
    gamma  : ndarray, Fourier series coefficient, params.exp_order square matrix
    zeta   : ndarray, Fourier series coefficient, params.exp_order square matrix
    x      : float, x_position
    y      : float, y-position 
    l_x    : float, length of simulation box in X-axis
    l_y    : float, length of simulation box in Y-axis

    OUPUTS
    h_x    : float, first order partial derivative by x
    h_y    : float, first order partial derivative by y
    h_xx   : float, second order partial derivative by x, x
    h_xy   : float, second order partial derivative by x, y
    h_yy   : float, second order partial derivative by y, y
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
            
            A = 2 * np.pi * n / l_x
            B = 2 * np.pi * m / l_y

            cosAx = np.cos(A * x)
            sinAx = np.sin(A * x)
            cosBy = np.cos(B * y)
            sinBy = np.sin(B * y)

            # First order derivatives
            h_x += (- alpha[n,m] * A * sinAx * cosBy
                    - beta[n,m]  * A * sinAx * sinBy
                    + gamma[n,m] * A * cosAx * cosBy
                    + zeta[n,m]  * A * cosAx * sinBy)

            h_y += (- alpha[n,m] * B * cosAx * sinBy
                    + beta[n,m]  * B * cosAx * cosBy
                    - gamma[n,m] * B * sinAx * sinBy
                    + zeta[n,m]  * B * sinAx * cosBy)

            # Second order derivatives
            h_xx += (- alpha[n,m] * A**2 * cosAx * cosBy
                     - beta[n,m]  * A**2 * cosAx * sinBy
                     - gamma[n,m] * A**2 * sinAx * cosBy
                     - zeta[n,m]  * A**2 * sinAx * sinBy)

            h_xy += (+ alpha[n,m] * B**2 * cosAx * cosBy
                     - beta[n,m]  * B**2 * cosAx * sinBy
                     - gamma[n,m] * B**2 * sinAx * cosBy
                     + zeta[n,m]  * B**2 * sinAx * sinBy)

            h_yy += (- alpha[n,m] * A * B * sinAx * sinBy
                     - beta[n,m]  * A * B * sinAx * cosBy
                     - gamma[n,m] * A * B * cosAx * sinBy
                     - zeta[n,m]  * A * B * cosAx * cosBy)

    return h_x, h_y, h_xx, h_xy, h_yy

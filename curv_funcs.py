# # # Imports # # # 
# Standard imports
import numpy as np
import matplotlib.pyplot as plt
# For Visualisation
import matplotlib.animation as animation
from scipy.interpolate import griddata


# Get manual input
exp_order = int(input('Order of Fourier series: '))


# # # Define Functions # # #

# Functions for height from Fourier series, its derivatives & the shape operator

def calc_height(alpha : np.array, beta : np.array, gamma : np.array, zeta : np.array, X : np.array, Y : np.array, l_x : float, l_y : float):
    '''
    Calculate height (z-direction) of model membrane using 2D Fourier expansion
    
    INPUT
    alpha  : ndarray, Fourier series coefficient, exp_order square matrix
    beta   : ndarray, Fourier series coefficient, exp_order square matrix
    gamma  : ndarray, Fourier series coefficient, exp_order square matrix
    zeta   : ndarray, Fourier series coefficient, exp_order square matrix
    x      : ndarray, x-position 
    y      : ndarray, y-postiion
    l_x    : float, length of simulation box in X-axis
    l_y    : float, length of simulation box in Y-axis
    
    OUPUT
    height : float, height at point (x,y)
    '''
    # Sum integers
    n = np.arange(exp_order)
    m = np.arange(exp_order)
    
    # Add two extra dimensions for broadcasting with 2D X, Y
    cos_nx = np.cos(2*np.pi*n[:, None, None]*X/l_x) 
    cos_my = np.cos(2*np.pi*m[:, None, None]*Y/l_y)  
    sin_nx = np.sin(2*np.pi*n[:, None, None]*X/l_x)
    sin_my = np.sin(2*np.pi*m[:, None, None]*Y/l_y)
    
    # Compute all four sums using einsum
    suma = np.einsum('nm,nij,mij->ij', alpha, cos_nx, cos_my)
    sumb = np.einsum('nm,nij,mij->ij', beta, cos_nx, sin_my)
    sumg = np.einsum('nm,nij,mij->ij', gamma, sin_nx, cos_my)
    sumz = np.einsum('nm,nij,mij->ij', zeta, sin_nx, sin_my)
    
    height = suma + sumb + sumg + sumz

    return height
    

def calc_fourier_derivatives(alpha : np.array, beta : np.array, gamma : np.array, zeta : np.array, X : np.array, Y : np.array, l_x : float, l_y : float):
    '''
    Compute a 2D Fourier expansion h(x,y) and its derivatives up to second order.

    INPUT
    alpha  : ndarray, Fourier series coefficients, exp_order square matrix
    beta   : ndarray, Fourier series coefficients, exp_order square matrix
    gamma  : ndarray, Fourier series coefficients, exp_order square matrix
    zeta   : ndarray, Fourier series coefficients, exp_order square matrix
    x      : ndarray, 2D array of x_positions (meshgrid)
    y      : ndarray, 2D array of y-positions (meshgrid)
    l_x    : float, length of simulation box in X-axis
    l_y    : float, length of simulation box in Y-axis

    OUPUTS
    h_x    : ndarray, first order partial derivative by x
    h_y    : ndarray, first order partial derivative by y
    h_xx   : ndarray, second order partial derivative by x, x
    h_xy   : ndarray, second order partial derivative by x, y
    h_yy   : ndarray, second order partial derivative by y, y
    '''
    # Sum integers
    n = np.arange(exp_order)
    m = np.arange(exp_order)
    
    # Compute differentiated trig arguments
    A = (2 * np.pi * n / l_x)[:, None, None]
    B = (2 * np.pi * m / l_y)[:, None, None]
    
    # Compute trig functions once
    cos_nx = np.cos(2*np.pi*n[:, None, None]*X/l_x)
    sin_nx = np.sin(2*np.pi*n[:, None, None]*X/l_x)
    cos_my = np.cos(2*np.pi*m[:, None, None]*Y/l_y)
    sin_my = np.sin(2*np.pi*m[:, None, None]*Y/l_y)
    
    # First order derivatives using einsum
    h_x = np.einsum('nm,nij,mij->ij', -alpha * (2*np.pi*n/l_x)[:, None], sin_nx, cos_my) + \
          np.einsum('nm,nij,mij->ij', -beta  * (2*np.pi*n/l_x)[:, None], sin_nx, sin_my) + \
          np.einsum('nm,nij,mij->ij', +gamma * (2*np.pi*n/l_x)[:, None], cos_nx, cos_my) + \
          np.einsum('nm,nij,mij->ij', +zeta  * (2*np.pi*n/l_x)[:, None], cos_nx, sin_my)
    
    h_y = np.einsum('nm,nij,mij->ij', -alpha * (2*np.pi*m/l_y)[None, :], cos_nx, sin_my) + \
          np.einsum('nm,nij,mij->ij', +beta  * (2*np.pi*m/l_y)[None, :], cos_nx, cos_my) + \
          np.einsum('nm,nij,mij->ij', -gamma * (2*np.pi*m/l_y)[None, :], sin_nx, sin_my) + \
          np.einsum('nm,nij,mij->ij', +zeta  * (2*np.pi*m/l_y)[None, :], sin_nx, cos_my)
    
    # Second order derivatives...
    # Compute differentiated trig arguments
    A_sq = (2*np.pi*n/l_x)**2
    B_sq = (2*np.pi*m/l_y)**2
    AB = (2*np.pi*n/l_x)[:, None] * (2*np.pi*m/l_y)[None, :]
    
    h_xx = np.einsum('nm,nij,mij->ij', -alpha * A_sq[:, None], cos_nx, cos_my) + \
           np.einsum('nm,nij,mij->ij', -beta  * A_sq[:, None], cos_nx, sin_my) + \
           np.einsum('nm,nij,mij->ij', -gamma * A_sq[:, None], sin_nx, cos_my) + \
           np.einsum('nm,nij,mij->ij', -zeta  * A_sq[:, None], sin_nx, sin_my)
    
    h_xy = np.einsum('nm,nij,mij->ij', +alpha * B_sq[None, :], cos_nx, cos_my) + \
           np.einsum('nm,nij,mij->ij', -beta  * B_sq[None, :], cos_nx, sin_my) + \
           np.einsum('nm,nij,mij->ij', -gamma * B_sq[None, :], sin_nx, cos_my) + \
           np.einsum('nm,nij,mij->ij', +zeta  * B_sq[None, :], sin_nx, sin_my)
    
    h_yy = np.einsum('nm,nij,mij->ij', -alpha * AB, sin_nx, sin_my) + \
           np.einsum('nm,nij,mij->ij', -beta  * AB, sin_nx, cos_my) + \
           np.einsum('nm,nij,mij->ij', -gamma * AB, cos_nx, sin_my) + \
           np.einsum('nm,nij,mij->ij', -zeta  * AB, cos_nx, cos_my)

    return h_x, h_y, h_xx, h_xy, h_yy


def calc_shape_operator(h_x : np.ndarray, h_y : np.ndarray, h_xx : np.ndarray, h_xy : np.ndarray, h_yy : np.ndarray):
    '''
    Calculate the shape operator at point (x,y)

    Note: in other code, -VE of shape operator used
    
    INPUT
    h_x    : ndarray, first order partial derivative by x
    h_y    : ndarray, first order partial derivative by y
    h_xx   : ndarray, second order partial derivative by x, x
    h_xy   : ndarray, second order partial derivative by x, y
    h_yy   : ndarray, second order partial derivative by y, y
    
    OUPUT
    S_xy   : 2D array, shape operator at point x,y
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

    return dA


# Functions for principle curvatures & descriptors

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
    K_G : float, Gaussian curvature# Get the number of dimensions in v

    OUPUT
    k_1 : float, principle curvature
    k_2 : float, principle curvature
    '''
    k_1 = H + np.sqrt( H**2 - K_G )
    k_2 = H - np.sqrt( H**2 - K_G )
    
    return k_1, k_2


def calc_angle(u : np.ndarray, v : np.ndarray):
    ''' 
    Compute angle between two vectors using dot product relation

    INPUTS
    u : ndarray, vector
    v : ndarray, vector, if longer must be 2nd

    OUPUT
    angle : float, angle in radians
    '''
    angle_rad = np.arccos( np.dot(v, u) / (np.linalg.vector_norm(u)*np.linalg.vector_norm(v, axis=-1)) )
    
    return angle_rad


def calc_curv_descriptors(S : np.ndarray):
    '''
    Calculate all curvature descriptors at point x, y
    *** Note: does not calculate principle vectors, required to calculate theta1 & 2
    
    INPUT
    S      : ndarray, shape operator at point x, y - 2x2 matrix 

    OUPUTS
    k1    : float, principle curvature 1
    k2    : float, principle curvature 2
    H     : float, mean curvature at point x,y
    K_G   : float, Gaussian curvature at point x, y
    '''
    # Curvature descriptors
    H   = calc_H(S)   # mean
    K_G = calc_K_G(S) # Gaussian

    # Principle curvatures
    k1, k2 = calc_principle_curvatures(H, K_G)

    return  k1, k2, H, K_G


def calc_curv_descriptors_full(S : np.ndarray):
    '''
    Calculate all curvature descriptors at point x, y
    
    INPUT
    S      : ndarray, shape operator at point x, y - 2x2 matrix of <npts>x<ntps> matrices

    OUPUTS
    k1     : float, principle curvature 1# Get the number of dimensions in v
    k2     : float, principle curvature 2
    H      : float, mean curvature at point x,y
    K_G    : float, Gaussian curvature at point x, y
    theta1 : float, angle (rad) between k1 and tangent to (flat) plane
    theta2 : float, angle (rad) between k2 and tangent to (flat) plane
    '''
    # Find eigenvalues (w) and [right] eigenfunctions (v) of shape operator
    S_transposed = np.transpose(S, (2, 3, 0, 1)) 
    w, v = np.linalg.eig(S_transposed)

    # Principle curvatures
    k1, k2 = w[:,:,0], w[:,:,1]     # 0th & 1st eigenvalues

    # Curvature descriptors
    H   = np.mean([k1,k2])  # mean
    K_G = k1*k2             # Gaussian
    
    # Principle vectors
    v1 = v[:,:,:,0]            
    v2 = v[:,:,:,1]
    ux = np.array([1, 0])   # vector tangental to (flat) plane

    # Angles to (flat) plane of principle vectors
    theta1 = calc_angle(ux, v1)
    theta2 = calc_angle(ux, v2)

    return  k1, k2, H, K_G, theta1, theta2


# Functions for curvature field tensors & weight function

def calc_mean_curv_tensor(k1 : float, k2 : float, theta : float):
    ''' 
    Calculate mean curvature field at point x,y

    INPUT
    k1    : float, principle curvature 1
    k2    : float, principle curvature 2
    theta : float, angle (rad) between k1 and tangent to (flat) plane

    OUPUT
    M : np.matrix, 2x2 matrix for mean curvature tensor at point x, y
    '''
    M = np.matrix(np.zeros((2,2)))
    
    M[0,0] = k1*np.cos(theta)**2 + k2*np.sin(theta)**2
    M[0,1] = (k2-k1) * np.cos(theta)*np.sin(theta)
    M[1,0] = (k2-k1) * np.cos(theta)*np.sin(theta)
    M[1,1] = k1*np.sin(theta)**2 + k2*np.cos(theta)**2   

    return M


def calc_spont_curv_tensor(t, phi):
    ''' 
    Calculate spontaneous curvature tensor at point x,y

    INPUT
    t   : ndarray, protein curvature tensor
    phi : float, angle (rad) between k1 and tangent to (flat) plane

    OUPUT
    C0 : np.matrix, 2x2 matrix for spontaneous curvature tensor at point x, y
    '''
    C0 = np.matrix(np.zeros((2,2)))
    
    C0[0,0] = t[0]*np.cos(phi)**2 + t[1]*np.sin(phi)**2
    C0[0,1] = (t[1]-t[0]) * np.cos(phi)*np.sin(phi)
    C0[1,0] = (t[1]-t[0]) * np.cos(phi)*np.sin(phi)
    C0[1,1] = t[0]*np.sin(phi)**2 + t[1]*np.cos(phi)**2
    
    return C0


def calc_wf(X : np.ndarray, Y : np.ndarray, p_com : np.ndarray, t : np.ndarray):
    '''
    Calculate weight function for decay of curvature tensor y-coord with distance from protein
    
    INPUTS
    X     : ndarray, x-position
    Y     : ndarray, y-position
    p_com : ndarray, protein centre of mass (com)
    t     : ndarray, protein curvature tensor, to be optimised

    OUTPUT
    wf    : ndarray, weight function for curvature tensor y-coord
    '''
    r  = np.sqrt((p_com[0] - X)**2 + (p_com[1] - Y)**2)
    wf = np.exp(-((r**2)/(t[2]**2)))
    
    return wf


# Functions for data (re)processing

def reformat_Fourier_coeffs(Fcoeffs):
    '''
    Reformat Fourier coefficients from 1D array (for scipy.optimize) to matrices of different coefficients

    INPUTS
    Fcoeffs : ndarray, Fourier series coefficients, shape 1x(4*exp_order**2) for scipy.optimize.least_squares

    OUTPUTS
    alpha  : ndarray, Fourier series coefficient, exp_order square matrix
    beta   : ndarray, Fourier series coefficient, exp_order square matrix
    gamma  : ndarray, Fourier series coefficient, exp_order square matrix
    zeta   : ndarray, Fourier series coefficient, exp_order square matrix
    '''
    # Format Fourier coefficients from 1D ndarray from scipy.optimize
    alpha = Fcoeffs[0*exp_order**2:1*exp_order**2].reshape(exp_order, exp_order)
    beta  = Fcoeffs[1*exp_order**2:2*exp_order**2].reshape(exp_order, exp_order)
    gamma = Fcoeffs[2*exp_order**2:3*exp_order**2].reshape(exp_order, exp_order)
    zeta  = Fcoeffs[3*exp_order**2:4*exp_order**2].reshape(exp_order, exp_order)

    return alpha, beta, gamma, zeta
    

# Putting functions together -- for protein tensor minimisation

def be_xy(alpha, beta, gamma, zeta, X, Y, l_x, l_y, p_com, pv, t):
    '''
    Put all functions together for loss calculation in minimisation step
    *** DEFINED IN NOTEBOOK FOR NOW ***

    INPUT
    alpha : ndarray, Fourier series coefficient, exp_order square matrix
    beta  : ndarray, Fourier series coefficient, exp_order square matrix
    gamma : ndarray, Fourier series coefficient, exp_order square matrix
    zeta  : ndarray, Fourier series coefficient, exp_order square matrix
    X     : ndarray, x-position
    Y     : ndarray, y-position
    l_x   : float, length of simulation box in X-axis
    l_y   : float, length of simulation box in Y-axis
    p_com : ndarray, protein centre of mass (com)
    pv    : ndarray, protein principle axis vector
    t     : ndarray, protein curvature tensor, to be optimised

    OUTPUT
    be    : ndarray, some function of curvature field
    '''
    pass # DEFINED IN NOTEBOOK FOR NOW


# Functions for optimisation/ objective function

def calc_residual(Fcoeffs : np.ndarray, h_sim : np.ndarray, X : np.ndarray, Y : np.ndarray, l_x : float, l_y : float):
    '''
    Residual for objective function in minimisation of difference of least squares

    INPUTS
    Fcoeffs : ndarray, Fourier series coefficients, shape 1x(4*exp_order**2) for scipy.optimize.least_squares
    h_sim   : ndarray, true height of simulated membrane over points X, Y
    X       : ndarray, X axis points
    Y       : ndarray, Y axis points
    l_x     : float, length of simulation box in X-axis
    l_y     : float, length of simulation box in Y-axis
 
    OUPUTS
    residual : float, residual cost of difference of least squares for Fourier surface to simulated membrane
    '''
    alpha, beta, gamma, zeta = reformat_Fourier_coeffs(Fcoeffs)

    residual = h_sim - calc_height( alpha, beta, gamma, zeta, X, Y, l_x, l_y )

    return residual[0] # reshape from (1, a) to (a,)


def tbe_g(t):
    '''
    Minimise parameter t for protein local curvature field
    *** DEFINED IN NOTEBOOK FOR NOW ***

    INPUT
    t : ndarray, protein curvature tensor

    OUTPUT
    obj_out : float, loss - objective function output for curvature tensor t
    '''
    pass # DEFINED IN NOTEBOOK FOR NOW


# Functions for visualisation

def visualise(Fcoeffs_data, X, Y, l_x, l_y, u_buckle, foldername):
    # Number of frames
    nframes = 10

    # Fourier coefficients & height
    Z_fit = []
    for i in range(len(Fcoeffs_data)):
        alpha, beta, gamma, zeta = reformat_Fourier_coeffs(Fcoeffs_data[i])
        Z_fit.append(calc_height(alpha, beta, gamma, zeta, X, Y, l_x, l_y))

    # True height
    Z_sim = []
    for idx, ts in enumerate(u_buckle.trajectory[:]):
        lipids = u_buckle.atoms.select_atoms('name PO4')
        pos = lipids.positions
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    
        # Interpolate scattered z values onto the same grid as Z_fit
        z_grid = griddata(
            (x, y), z, (X, Y),
            method='cubic',  # or 'linear' if cubic fails
            fill_value=np.nan
        )
        Z_sim.append(z_grid)

    # Compute color scale ranges
    Z_min_fit, Z_max_fit = np.min(Z_fit), np.max(Z_fit)
    Z_min_sim, Z_max_sim = np.nanmin(Z_sim), np.nanmax(Z_sim)
    maxval = max(abs(Z_min_fit), abs(Z_max_fit))
    #maxval /= 100

    # Create dummy Z for colorbar scale
    Z_for_colourbar = np.copy(Z_fit[0])
    Z_for_colourbar[0, 0], Z_for_colourbar[0, 1] = np.min([Z_min_fit, Z_min_sim]), np.max([Z_max_fit, Z_max_fit])
    
    # Create subplots 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Initial contour plots
    contour1 = ax1.contourf(X, Y, Z_for_colourbar, levels=50, cmap='viridis', vmin=-maxval, vmax=maxval)
    contour2 = ax2.contourf(X, Y, Z_for_colourbar, levels=50, cmap='viridis', vmin=-maxval, vmax=maxval)

    # Shared colorbar
    cbar = fig.colorbar(contour2, ax=[ax1,ax2])
    cbar.set_label("Height")

    # Labels and titles
    ax2.set_title("Reconstructed (Frame 0)")
    ax1.set_title("Simulated (Frame 0)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax2.set_xlabel("X")

    # Update function for animation
    def update(frame):
        nonlocal contour1, contour2

        # Remove previous contour
        if contour1 is not None:
            contour1.remove()  # -- for matplotlib >= 3.8
        if contour2 is not None:
            contour2.remove()

        # Update contour plots
        contour1 = ax1.contourf(X, Y, Z_sim[frame], levels=50, cmap='viridis', vmin=-maxval, vmax=maxval)
        contour2 = ax2.contourf(X, Y, Z_fit[frame], levels=50, cmap='viridis', vmin=-maxval, vmax=maxval)

        # Update titles
        ax2.set_title(f"Reconstructed — Frame {frame}")
        ax1.set_title(f"Simulated — Frame {frame}")

        return [contour1, contour2]

    # Create and save animation
    anim = animation.FuncAnimation(fig, update, frames=len(Z_fit), interval=100, blit=False)
    anim.save(f'./{foldername}/sim_vs_fit.gif', writer=animation.PillowWriter(fps=10))
    plt.show()

    return anim
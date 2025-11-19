# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 12:35:32 2025

@author: ralph-holden

Script containing functions for Markov chain Monte Carlo simulation of membrane geometry
Uses Helfrich bending energy for membrane surface, parametrised by Fourier series coefficients and over descrete grid, to evaluate Monte Carlo moves
"""
# # # Imports # # #
# For simulatution
import numpy as np
import copy
# For visualisation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D



# # # Information storing classes # # #

class params:
    '''
    Set parameters used in simulation
    '''
    # Size of thermal fluctuations, base unit of simulation energy
    kbT = 1  
    
    # Box size
    l_x = 50                     # box size, x-direction
    l_y = 50                     # box size, y-direction
    
    # Fourier expansion
    exp_order = 4                # order of 2D Fourier expansion
    
    # Bending energies
    H_0     = 0.0                # Spontaneous mean curvature
    kappa_H = 20.0               # Bending rigidity of mean curvature (kbT*l units)
    kappa_K = -20.0              # Bending rigidity of Gaussian curvature (kbT*l^2 units)
    
    # Size of Monte Carlo moves
    delta = 0.05                 # Standard deviation of perturbation applied to Fourier coefficients
    
    # Maximum change in projected area
    original_excess_area = 0     # placeholder for later calculation
    dA_threshold         = 0.05  # Fraction change allowed from starting membrane
    
    # X, Y grid for calculations
    npts = 5                     # Number of points per l unit length -> npts^2 per l^2 unit area
    x = np.linspace(0, l_x, l_x * npts)
    y = np.linspace(0, l_y, l_y * npts)
    X, Y = np.meshgrid(x, y)



# # # Functions for calculating Helfrich bending energy and other parameters # # #

def calc_height(membrane : dict, X : np.array, Y : np.array):
    '''
    Calculate height (z-direction) of model membrane using 2D Fourier expansion
    
    INPUT
    membrane : dict, contains Fourier coefficients (params.exp_order square matrices) for flat surface and the associated bending energy
    x        : ndarray, 2D array of x_positions (meshgrid)
    y        : ndarray, 2D array of y-positions (meshgrid)
    
    OUPUT
    height   : float, height at point (x,y)
    '''
    # Sum integers
    n = np.arange(params.exp_order)
    m = np.arange(params.exp_order)
    
    # Add two extra dimensions for broadcasting with 2D X, Y
    cos_nx = np.cos(2*np.pi*n[:, None, None]*X/params.l_x) 
    cos_my = np.cos(2*np.pi*m[:, None, None]*Y/params.l_y)  
    sin_nx = np.sin(2*np.pi*n[:, None, None]*X/params.l_x)
    sin_my = np.sin(2*np.pi*m[:, None, None]*Y/params.l_y)
    
    # Compute all four sums using einsum
    suma = np.einsum('nm,nij,mij->ij', membrane['alpha'], cos_nx, cos_my)
    sumb = np.einsum('nm,nij,mij->ij', membrane['beta'],  cos_nx, sin_my)
    sumg = np.einsum('nm,nij,mij->ij', membrane['gamma'], sin_nx, cos_my)
    sumz = np.einsum('nm,nij,mij->ij', membrane['zeta'],  sin_nx, sin_my)
    
    height = suma + sumb + sumg + sumz

    return height
    

def calc_fourier_derivatives(membrane : dict, X : np.array, Y : np.array):
    '''
    Compute a 2D Fourier expansion h(x,y) and its derivatives up to second order.

    INPUT
    membrane : dict, contains Fourier coefficients (params.exp_order square matrices) for flat surface and the associated bending energy
    x        : ndarray, 2D array of x_positions (meshgrid)
    y        : ndarray, 2D array of y-positions (meshgrid)

    OUPUTS
    h_x      : float, first order partial derivative by x
    h_y      : float, first order partial derivative by y
    h_xx     : float, second order partial derivative by x, x
    h_xy     : float, second order partial derivative by x, y
    h_yy     : float, second order partial derivative by y, y
    '''
    # Sum integers
    n = np.arange(params.exp_order)
    m = np.arange(params.exp_order)

    # Precompute differentiated trig arguments
    A = 2 * np.pi * n / params.l_x
    B = 2 * np.pi * m / params.l_y
    
    # Precompute trig functions
    cos_nx = np.cos(2*np.pi*n[:, None, None]*X/params.l_x)
    sin_nx = np.sin(2*np.pi*n[:, None, None]*X/params.l_x)
    cos_my = np.cos(2*np.pi*m[:, None, None]*Y/params.l_y)
    sin_my = np.sin(2*np.pi*m[:, None, None]*Y/params.l_y)
    
    # First order derivatives using einsum
    h_x = np.einsum('nm,nij,mij->ij', -membrane['alpha'] * A[:, None], sin_nx, cos_my) + \
          np.einsum('nm,nij,mij->ij', -membrane['beta']  * A[:, None], sin_nx, sin_my) + \
          np.einsum('nm,nij,mij->ij', +membrane['gamma'] * A[:, None], cos_nx, cos_my) + \
          np.einsum('nm,nij,mij->ij', +membrane['zeta']  * A[:, None], cos_nx, sin_my)
    
    h_y = np.einsum('nm,nij,mij->ij', -membrane['alpha'] * B[None, :], cos_nx, sin_my) + \
          np.einsum('nm,nij,mij->ij', +membrane['beta']  * B[None, :], cos_nx, cos_my) + \
          np.einsum('nm,nij,mij->ij', -membrane['gamma'] * B[None, :], sin_nx, sin_my) + \
          np.einsum('nm,nij,mij->ij', +membrane['zeta']  * B[None, :], sin_nx, cos_my)
    
    # Second order derivatives...
    # Precompute differentiated trig arguments
    A_sq = A**2
    B_sq = B**2
    AB = A[:, None] * B[None, :]
    
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
    Calculate the area element, dA -- increase in surface area of bent surface at gridpoint x,y
    Note: x, y positions specified when calculating h_x, h_y in calc_shape_operator 

    INPUTS
    h_x : float, partial derivative of height by x at point x,y
    h_y : float, partial derivative of height by y at point x,y

    OUPUTS
    dA  : float, area element
    '''
    dA = np.sqrt(1 + h_x**2 + h_y**2)

    return dA


def calc_H(S_xy: np.ndarray):
    '''
    Calculate mean curvature, H at gridpoint(s) x,y
    
    INPUT
    S_xy : 2D array, shape operator at gridpoint(s) x,y
    
    OUPUT
    H    : float, mean curvature
    '''
    H = 0.5 * np.trace(S_xy)
    
    return H


def calc_K_G(S_xy: np.ndarray):
    '''
    Calculate Gaussian curvature, K_G at gridpoint(s) x,y
    When S_xy is over all of grid X,Y -- must make determinant piecewise
    
    INPUT
    S_xy : 2D array, shape operator at gridpoint(s) x,y
    
    OUPUT
    K_G  : float, Gaussian curvature at gridpoint(s) x,y
    '''
    #K_G = np.linalg.det(S_xy) # does not work for all gridpoints embedded in S_xy
    a = S_xy[0, 0] 
    b = S_xy[0, 1]
    c = S_xy[1, 0]
    d = S_xy[1, 1]
    # Compute determinant
    K_G = a * d - b * c 
    
    return K_G


def calc_principle_curvatures(H : float, K_G : float):
    '''
    Calculate principle curvatures k_1, k_2 at gridpoint(s) x,y
    
    INPUT
    H   : float, mean curvature at gridpoint(s) x,y
    K_G : float, Gaussian curvature at gridpoint(s) x,y
    
    OUPUT
    k_1 : float, principle curvature at gridpoint(s) x,y
    k_2 : float, principle curvature at gridpoint(s) x,y
    '''
    k_1 = H + np.sqrt( H**2 - K_G )
    k_2 = H - np.sqrt( H**2 - K_G )
    
    return k_1, k_2


def calc_Helfrich_energy(H : np.ndarray, K_G : np.ndarray, dA = 1):
    '''
    Calculate Helfrich bending energy of membrane, for energy change in Monte Carlo step
    Membrane surface parametrised by Fourier coefficients, on descrete grid X,Y
    Using change in 2D area from bending, dA, to rescale the 2D area at every point
    
    INPUT
    H          : ndarray, mean curvature at gridpoint(s) x,y
    K_G        : ndarray, Gaussian curvature at gridpoint(s) x,y
    dA         : ndarray, area element at gridpoint(s) x,y -- increase in surface area from membrane bending 
    
    OUTPUT
    tot_energy : float, Helfrich bending energy over surface
    '''
    # Calculate Helfrich bending energy for each point in surface
    energy_per_pnt = 2*params.kappa_H * ( H - params.H_0 )**2 + abs(params.kappa_K * K_G) # abs to avoid negative bending energy

    # Calculate subgrid area per point, with dA correction from bending
    subgrid_area   = 1 / (dA * params.npts**2)

    # Riemann integral of energy over surface area, truncated by subgrid area
    tot_energy     = np.sum(energy_per_pnt * subgrid_area)
    
    return tot_energy



# # # Functions for Markov Chain Monte Carlo # # #

def montecarlosubmove(prev_membrane : dict):
    '''
    First step of Markov Chain Monte Carlo (MCMC) move to Fourier coefficients describing surface
    Proposes new membrane with change to previous membrane and calculates excess area, but does NOT test for excess area criterion or calculate bending energy
    
    INPUT
    prev_membrane : dict, contains curvature Fourier coefficients and energy from previous step
    
    OUTPUT
    move_membrane      : dict, new membrane with perturbed curvature Fourier coefficients WITHOUT associated energy and NOT vetted for dA criterion
    S                  : ndarray, shape operator for all grid points X, Y
    dA                 : ndarray, total surface area from membrane bending for all grid points X, Y
    excess_area_change : float, fraction change in excess area from initial membrane
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

    # Calculate change in area from membrane bending (for each subgrid point)
    dA = calc_area_element(h_x, h_y)

    # Calculate excess area
    excess_area = np.sum(dA) / params.npts**2 - params.l_x*params.l_y
    # Calculate fraction change from original membrane for excess area criterion
    excess_area_change = abs( (excess_area-params.original_excess_area) / params.original_excess_area ) if params.original_excess_area!=0 else 0
    # Note: if original excess area is 0, excess_area_change set to 0 -> no excess area criterion
    
    return move_membrane, S, dA, excess_area_change


def montecarlomove(prev_membrane : dict):
    '''
    Entire of Markov Chain Monte Carlo (MCMC) move to Fourier coefficients describing surface
    Additional excess area criterion: need to make sure move does not change projected area beyond a threshold 
    -- thereby maintaining an initial buckle shape
    
    INPUT
    prev_membrane : dict, contains curvature Fourier coefficients and energy from previous step
    
    OUTPUT
    move_membrane : dict, new (& vetted) membrane with perturbed curvature Fourier coefficients AND associated energy
    '''
    # Propose Markov Chain Monte Carlo 
    move_membrane, S, dA, excess_area_change = montecarlosubmove(prev_membrane)

    # Ensure Monte Carlo Move does not deviate too far from original excess area
    while excess_area_change > params.dA_threshold:
        
        #print('Exceeded dA threshold with fraction excess area change', excess_area_change, 'from original', params.original_excess_area)
        
        # Propose new Markov Chain Monte Carlo 
        move_membrane, S, dA, excess_area_change = montecarlosubmove(prev_membrane)
        
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
    Evaluate move using Monte Carlo criterion
    
    INPUT
    move_energy : float, energy associated with proposed membrane curvature
    prev_energy : float, energy associated with previous membrane curvature
    
    OUPUT
    accept_move : bool, Monte Carlo step outcome
    '''
    # Calculate Boltzmann factor of Helfrich bending energy change
    boltzmann_factor = np.exp( -(move_energy - prev_energy) )
    
    # Draw random number from uniform distribution between [0,1)
    rand_number = np.random.random()

    # Monte Carlo acceptance criterion
    accept_move = rand_number < boltzmann_factor
    
    return accept_move


def montecarlostep(membrane_lst : list):
    '''
    Run MCMC step; propose move, evaluate, update Markov chain
    
    INPUT
    membrane_lst : list of dict, contains curvature Fourier coefficients and associated energy from previous steps
    
    OUPUT
    membrane_lst : list of Model_membrane, UPDATED ensemble of curvature Fourier coefficients and associated energy
    accept_move  : bool, Monte Carlo step outcome
    '''
    # Extract last state data
    prev_membrane = membrane_lst[-1]
    
    # Make Markov chain Monte Carlo move, subject to excess area criterion
    move_membrane = montecarlomove(prev_membrane)

    # Extract bending energies from dictionaries
    prev_energy = prev_membrane['energy']
    move_energy = move_membrane['energy']
    
    # Evaluate
    accept_move = montecarloeval(move_energy, prev_energy)
    
    # Update Markov chain
    membrane_lst += [move_membrane] if accept_move else [prev_membrane]

    return membrane_lst, accept_move



# # # Function to calculate starting membrane data # # #

def init_model_membrane( membrane = None ):
    '''
    Make class that stores Fourier cofficients describing surface (for 3rd order 2D expansion)

    INPUTS
    membrane : dict or None, contains exp_order square matrices for initial Fourier coefficients,
               Default - None - initial membrane is flat

    OUTPUT
    membrane : dict, contains Fourier coefficients for flat surface and the associated bending energy
    ''' 
    if not membrane:
        membrane = {'alpha' : np.zeros((params.exp_order, params.exp_order)),
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

    # Calculate total surface area
    total_area  = np.sum(dA) / params.npts**2
    # Excess area is total surface area minus projected (2D) area
    excess_area = total_area - params.l_x*params.l_y
    # Save to params for montecarlostep criterion
    params.original_excess_area = excess_area

    # Print attributes
    print('Initial membrane bending energy:', bending_energy, 'kbT , excess area:', excess_area)
    
    return membrane

    

# # # Code for visualisation / data analysis # # #

class visualise:

    def __init__(self, membrane_lst : list, nframes : int, save_dir=''):
        '''
        Make calculations prior to animating plots

        INPUTS
        membrane_lst : list of dict, contains curvature Fourier coefficients and associated energy
        nframes      : int, frequency/ interval size of frames
        save_dir     : str, name of directory to save animation, from view of working directory
        '''
        self.nframes  = nframes
        self.save_dir = save_dir
        
        # X,Y grid
        npts = 4 
        x = np.linspace(0, params.l_x, params.l_x*npts)
        y = np.linspace(0, params.l_y, params.l_y*npts)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Calculate z-direction (heights)
        self.Z_dump = [calc_height(membrane, self.X, self.Y) for membrane in membrane_lst[::nframes]]
        
        # Extract associated bending energies
        self.energy_lst = [membrane['energy'] for membrane in membrane_lst[::nframes]]
        
        # Find that with maximum range for colourbar
        Z_min, Z_max = np.min(self.Z_dump), np.max(self.Z_dump)
        self.maxval = Z_max if Z_max>abs(Z_min) else abs(Z_min)
        self.Z_for_colourbar = self.Z_dump[0] + 0.0 # trick to make copy
        self.Z_for_colourbar[0], self.Z_for_colourbar[1] = Z_min, Z_max # edit values so they show up on colourbar


    def vis_contour(self):
        '''
        Visualise membrane curvature ensemble using heatmap plots
        Note: un/comment code for use between matplotlib versions < & > 3.8 
        
        OUPUT
        anim         : matplotlib.animation.FuncAnimation, animation of heatmap plots of membrane height
        '''
        # Animation plot
        fig, ax = plt.subplots(figsize=[8, 6])
        
        # Initial contour and colorbar
        contour = ax.contourf(self.X, self.Y, self.Z_for_colourbar, levels=50, cmap='viridis', vmin=-self.maxval, vmax=self.maxval)
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
                self.X, self.Y, self.Z_dump[frame],
                levels=50, cmap='viridis',
                vmin=-abs(self.maxval), vmax=+abs(self.maxval))
        
            # Update title
            title.set_text(f"Step: {frame * self.nframes} , Energy: {round(self.energy_lst[frame], 1)}")
    
            #return contour.collections + [title] # -- for matplotlib <3.8
            return [contour, title] 
            
        anim = animation.FuncAnimation(fig, update, frames=len(self.Z_dump), interval=100, blit=False, repeat=True)
        anim.save(f'./{self.save_dir}/contour_animation.gif', writer=animation.PillowWriter(fps=10))
        anim.save(f'./{self.save_dir}/contour_animation.mp4', writer='ffmpeg', fps=10)
        plt.show()
        
        return anim


    def vis_3d(self):
        '''
        Visualise membrane curvature ensemble using animation of 3D surface plots
        
        OUPUT
        anim         : matplotlib.animation.FuncAnimation, animation of heatmap plots of membrane height
        '''
        # Create figure and 3D axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize the surface plot
        surf = ax.plot_surface(self.X, self.Y, self.Z_for_colourbar, cmap='viridis', edgecolor='none', alpha=0.7, vmin=-self.maxval, vmax=self.maxval)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
        ax.set_title('3D Surface Animation')
        
        # Set fixed z-axis limits for consistent viewing
        ax.set_zlim(-self.maxval, self.maxval)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Animation update function
        def update(frame):
            ax.clear()
            
            # Plot the surface
            surf = ax.plot_surface(self.X, self.Y, self.Z_dump[frame], cmap='viridis', edgecolor='none', alpha=0.9)
            
            # Reset labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Height')
            ax.set_title(f"Step: {frame * self.nframes} , Energy: {round(self.energy_lst[frame], 1)}")
            ax.set_zlim(-self.maxval, self.maxval)
            
            return surf,
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(self.Z_dump), interval=100, blit=False, repeat=True)
        
        plt.tight_layout()
        anim.save(f'./{self.save_dir}/surface_animation.gif', writer='pillow', fps=10)
        anim.save(f'./{self.save_dir}/surface_animation.mp4', writer='ffmpeg', fps=10)
        plt.show()
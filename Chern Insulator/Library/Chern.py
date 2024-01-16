import numpy as np
import matplotlib.pyplot as plt
import math 
from mpl_toolkits.mplot3d import Axes3D


### 2D CHern Model_____________________________________________________________________________________________________________
def Ham_2D(k_x, k_y,m1, t1):
    """This function returns the Hamiltonian of the 3D Chern model.

    Args:
        k_x (float): Momentum in x direction
        k_y (float): Momentum in y direction
        t1 (float): hopping parameter for sheet 1
        m1 (float): onsite energy for sheet 1
    Returns:
        array: The Hamiltonian of the 3D Chern model
    """
    d_1 = np.sin(k_x)
    d_2 = np.sin(k_y)
    d_3 = m1 - 2*t1*(np.cos(k_x) + np.cos(k_y))
    d_5 = d_1 - 1j*d_2
    matrix = [[d_3, d_5],[np.conjugate(d_5),-d_3]]
    return matrix


## Band Structure_______________________________________________________________________ 
def plot_eigenvalues_2D(m1, t1):
    """This function plots the eigenvalues of the 3D Chern model for a fixed k_z value.
    """
    k_points = 200
    k_x_values = np.linspace(-np.pi, np.pi, k_points)
    k_y_values = np.linspace(-np.pi, np.pi, k_points)
    k_x_mesh, k_y_mesh = np.meshgrid(k_x_values, k_y_values)
    eigenvalues = np.zeros((k_points, k_points, 2))

    # Define a list of color maps for each band
    color_maps = ['cool', 'hot']

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(k_points):
        for j in range(k_points):
            k_x = k_x_mesh[i, j]
            k_y = k_y_mesh[i, j]
            matrix = Ham_2D(k_x, k_y,m1, t1)

            eigenvalues[i, j] = np.linalg.eigvalsh(matrix)

    for band in range(2):
        eigenvalues_band = eigenvalues[:, :, band]

        # Use a unique color map for each band
        cmap = plt.get_cmap(color_maps[band])

        # Create a surface plot with the specified color map
        ax.plot_surface(k_x_mesh, k_y_mesh, eigenvalues_band, cmap=cmap)
# set figure size
    fig = plt.figure(figsize=(6, 6))
    ax.set_xlabel('$k_x$', fontsize=14)
    ax.set_ylabel('$k_y$', fontsize=14)
    ax.set_zlabel('Energy')
    # Set the title to include parameters and fixed momentum index
    title = f"Parameters: t={t1}, m={m1}"
    ax.set_title(title, fontsize=14)
    # Enable interactive rotation
    ax.view_init(elev=5, azim=15)
    plt.show()

## d(k) surface plot_____________________________________________
# The surface of the Model Hamiltonian in d_x, d_y, d_z plane
def surface(k_x,k_y,m,t):
    """The eigenvectors of the model Hamiltonian for the QWZ model.

    Args:
        k_x (float): momementum in x direction
        k_y (float): momentum in y direction
        m (float): onsite energy
        t (float): hopping term

    Returns:
        float: The surface d(k) of the Model Hamiltonain
    """
    d_x = np.sin(k_x)
    d_y = np.sin(k_y)
    d_z = m - 2*t*(np.cos(k_x) + np.cos(k_y))
    return d_x, d_y, d_z

def sur_plot(t1, t2):
    k1 = np.linspace(-np.pi, np.pi, 200)
    k2 = np.linspace(-np.pi, np.pi, 200)
    x, y = np.meshgrid(k1, k2)
    d_1 = surface(x, y, t1, t2)[0]
    d_2 = surface(x, y, t1, t2)[1]
    d_3 = surface(x, y, t1, t2)[2]
    # Plot the surface
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(d_1, d_2, d_3, cmap='viridis',alpha = 0.8)

    ax.set_xlabel('d_x', fontsize=16)
    ax.set_ylabel('d_y', fontsize=16)
    ax.set_zlabel('d_z', fontsize=16)
    
    # Set axis limits
    #ax.set_xlim(0, 1.0)  # Set the x-axis limits
    #ax.set_ylim(-1.0, 1.0)  # Set the y-axis limits
    #ax.set_zlim(-2.0, 2.0)  # Set the z-axis limits
    
    # Add a dot at (0, 0, 0)
    ax.scatter(0, 0, 0, color='red', s=300, label='Origin (0, 0, 0)')
    
    #plt.legend(loc='upper right')
    plt.title('The surface d(k) of the Model Hamiltonain with m = {} and t = {}'.format(t1, t2))
    # Enable interactive rotation
    ax.view_init(elev=25, azim=45)  
    plt.show()
    
# check if the torus crosses the origin or not 
def check(m, t):
    """This function checks if the surface of the d(k) crosses the origin or not.

    Args:
        m (float): on-site energy
        t (float): hopping term

    Returns:
        str: Information about whether the surface crosses the origin and the values of m and t.
    """
    k_x = np.linspace(-np.pi, np.pi, 200)
    k_y = np.linspace(-np.pi, np.pi, 200)
    d_z = m - 2 * t * (np.cos(k_x) + np.cos(k_y))
    
    # Check if the origin (0, 0, 0) is crossed by the surface
    has_positive = np.any(d_z > 0)
    has_negative = np.any(d_z < 0)
    
    if has_positive and has_negative:
        return "The surface crosses the origin for m = {} and t = {} i.e Chern number is non-zero.".format(m, t)
    else:
        return "The surface does not cross the origin for m = {} and t = {} i.e Chern number is zero.".format(m, t)

## Chern Number Calculation_____________________________________________
def calculate_berry_(eigenvectors, band_index):
    """Calculate the Berry flux 

    Args:
        eigenvectors (array): Array of eigenvectors.
        band_index (int): Index of the band for which Berry curvature is calculated.
    Returns:
        array: Berry curvature array.
        float: Total flux.
    """
    N = len(eigenvectors)
    berry_flux = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            band1_00 = eigenvectors[i, j, :, band_index]
            band1_10 = eigenvectors[(i + 1) % N, j, :, band_index]
            band1_01 = eigenvectors[i, (j + 1) % N, :, band_index]
            band1_11 = eigenvectors[(i + 1) % N, (j + 1) % N, :, band_index]

            phase1 = np.dot(band1_00, np.conjugate(band1_10)) / np.linalg.norm(np.dot(band1_00, np.conjugate(band1_10)))
            phase2 = np.dot(band1_10, np.conjugate(band1_11)) / np.linalg.norm(np.dot(band1_10, np.conjugate(band1_11)))
            phase3 = np.dot(band1_11, np.conjugate(band1_01)) / np.linalg.norm(np.dot(band1_11, np.conjugate(band1_01)))
            phase4 = np.dot(band1_01, np.conjugate(band1_00)) / np.linalg.norm(np.dot(band1_01, np.conjugate(band1_00)))

            phaset = phase1 * phase2 * phase3 * phase4

            berry_flux[i, j] = np.angle(phaset)

    return berry_flux

def chern_number(n, band_index):
    # Define the range of kx and ky values
    k_values = np.linspace(-np.pi, np.pi, n)
    kx, ky = np.meshgrid(k_values, k_values)

    # Calculate the eigenvalues for each combination of kx and ky
    eigenvalues = np.zeros((n, n, 2))
    eigenvectors = np.zeros((n, n, 2, 2), dtype=np.complex64)
    for i in range(len(k_values)):
        for j in range(len(k_values)):
            # put the m and t values here
            H = Ham_2D(kx[i, j], ky[i, j], -3, 1)
            eigenvalues[i, j], eigenvectors[i, j] = np.linalg.eigh(H)

    # Calculate Berry curvature
    berry_flux = calculate_berry_(eigenvectors, band_index)

    # Print results
    chern_number = np.sum(berry_flux) / (2 * np.pi)
    print('Chern number: ', chern_number)



### 3D Chern Model ___________________________________________________________________________________________________
def Ham_3D(k_x, k_y, k_z, m1, t1, m2,t2, a, b, c, d):
    """This function returns the Hamiltonian of the 3D Chern model.

    Args:
        k_x (float): Momentum in x direction
        k_y (float): Momentum in y direction
        k_z (float): Momentum in z direction
        t1 (float): hopping parameter for sheet 1
        m1 (float): onsite energy for sheet 1
        t2 (float): hopping parameter for sheet 2
        m2 (float): onsite energy for sheet 2
        a (float): up to up orbital hopping parameter from sheet 1 to sheet 2
        b (float): up to down orbital hopping parameter from sheet 1 to sheet 2
        c (float): down to up orbital hopping parameter from sheet 1 to sheet 2
        d (float): down to down orbital hopping parameter from sheet 1 to sheet 2

    Returns:
        array: The Hamiltonian of the 3D Chern model
    """
    d_1 = np.sin(k_x)
    d_2 = np.sin(k_y)
    d_3 = m1 - 2*t1*(np.cos(k_x) + np.cos(k_y))
    d_4 = m2- 2*t2*(np.cos(k_x) + np.cos(k_y))
    d_5 = d_1 - 1j*d_2
    c_1 = np.exp(-1j*k_z)*a
    c_2 = np.exp(-1j*k_z)*b
    c_3 = np.exp(-1j*k_z)*c
    c_4 = np.exp(-1j*k_z)*d
    matrix = [[d_3, d_5,c_1,c_2],[np.conjugate(d_5),-d_3, c_3,c_4],[np.conjugate(c_1),np.conjugate(c_3),d_4, d_5],
              [np.conjugate(c_2),np.conjugate(c_4),np.conjugate(d_5),-d_4]]
    return matrix

## Band Structure_______________________________________________________________________
# Plot the band structure for a fixed k_z value
def plot_eigenvalues(N, kz,m1, t1, m2, t2, a, b, c, d):
    # Define the range of kx and ky values
    k_values = np.linspace(-np.pi, np.pi, N)
    kx, ky = np.meshgrid(k_values, k_values)

    # Calculate the eigenvalues for each combination of kx and ky
    eigenvalues = np.zeros((N, N, 4))
    for i in range(len(k_values)):
        for j in range(len(k_values)):
            H = Ham_3D(kx[i, j], ky[i, j], kz, m1, t1, m2, t2, a, b, c, d)
            eigenvalues[i, j], _ = np.linalg.eigh(H)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the eigenvalues
    for band in range(4):
        ax.plot_surface(kx, ky, eigenvalues[:, :, band], label=f'Band {band + 1}')

    # Set labels and title
    ax.set_xlabel(r'$k_x$', fontsize=16)
    ax.set_ylabel(r'$k_y$', fontsize=16)
    ax.set_zlabel('Energy', fontsize=16)
    ax.set_title('Eigenvalues of Hamiltonian')

    # Change the point of view (elevation, azimuth)
    ax.view_init(elev=0, azim=-65)

    # Adjust the fontsize of the axis markings
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(18)  # Change the fontsize for x-axis tick labels
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)  # Change the fontsize for y-axis tick labels
    for tick in ax.zaxis.get_major_ticks():
        tick.label1.set_fontsize(18)  # Change the fontsize for z-axis tick labels

    # Increase the distance between tick labels and xlabel and ylabel
    ax.xaxis.set_tick_params(pad=0)  # Increase the distance for x-axis tick labels
    ax.yaxis.set_tick_params(pad=0)  # Increase the distance for y-axis tick labels
    ax.zaxis.set_tick_params(pad=0)

    # Show the plot
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()


## Chern NUMBER CALCULATION_______________________________________________________________________
def calculate_berry_(eigenvectors, band_index):
    """Calculate the Berry flux 

    Args:
        eigenvectors (array): Array of eigenvectors.
        band_index (int): Index of the band for which Berry curvature is calculated.
    Returns:
        array: Berry curvature array.
        float: Total flux.
    """
    N = len(eigenvectors)
    berry_flux = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            band1_00 = eigenvectors[i, j, :, band_index]
            band1_10 = eigenvectors[(i + 1) % N, j, :, band_index]
            band1_01 = eigenvectors[i, (j + 1) % N, :, band_index]
            band1_11 = eigenvectors[(i + 1) % N, (j + 1) % N, :, band_index]

            phase1 = np.dot(band1_00, np.conjugate(band1_10)) / np.linalg.norm(np.dot(band1_00, np.conjugate(band1_10)))
            phase2 = np.dot(band1_10, np.conjugate(band1_11)) / np.linalg.norm(np.dot(band1_10, np.conjugate(band1_11)))
            phase3 = np.dot(band1_11, np.conjugate(band1_01)) / np.linalg.norm(np.dot(band1_11, np.conjugate(band1_01)))
            phase4 = np.dot(band1_01, np.conjugate(band1_00)) / np.linalg.norm(np.dot(band1_01, np.conjugate(band1_00)))

            phaset = phase1 * phase2 * phase3 * phase4

            berry_flux[i, j] = np.angle(phaset)

    return berry_flux

def chern_number_3D(n, band_index):
    # Define the range of kx and ky values
    k_values = np.linspace(-np.pi, np.pi, n)
    kx, ky = np.meshgrid(k_values, k_values)

    # Calculate the eigenvalues for each combination of kx and ky
    eigenvalues = np.zeros((n, n, 4))
    eigenvectors = np.zeros((n, n, 4, 4), dtype=np.complex64)
    for i in range(len(k_values)):
        for j in range(len(k_values)):
            H = Ham_3D(kx[i, j], ky[i, j], 0, 3, 1, -6, 1,0, 1j, 0, 0)
            eigenvalues[i, j], eigenvectors[i, j] = np.linalg.eigh(H)

    # Calculate Berry curvature
    berry_flux = calculate_berry_(eigenvectors, band_index)

    # Print results
    chern_number = np.sum(berry_flux) / (2 * np.pi)
    print('Chern number: ', chern_number)

##Plottting Chern Number_______________________________________________________________________
def Plot_chern_number(param_values, n, band_index):
    chern_numbers = []

    for param in param_values:
        k_values = np.linspace(-np.pi, np.pi, n)
        kx, ky = np.meshgrid(k_values, k_values)

        eigenvalues = np.zeros((n, n, 4))
        eigenvectors = np.zeros((n, n, 4, 4), dtype=np.complex64)

        for i in range(len(k_values)):
            for j in range(len(k_values)):
                H = Ham_3D(kx[i, j], ky[i, j], 0, 3, 1, -6, 1,0, param, 0, 0)
                eigenvalues[i, j], eigenvectors[i, j] = np.linalg.eigh(H)

        berry_flux = calculate_berry_(eigenvectors, band_index)
        chern_number = np.sum(berry_flux) / (2 * np.pi)
        chern_numbers.append(chern_number)

    plt.plot(param_values, chern_numbers, marker='o')
    # set the marker size
    plt.rcParams['lines.markersize'] = 5
    plt.xlabel('b')
    plt.ylabel('Chern Number')
    # set y axis limit
    plt.ylim(-4, 4)
    plt.title('Chern Number for u = 3, v = -6')
    plt.show()
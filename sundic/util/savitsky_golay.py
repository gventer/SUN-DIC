################################################################################
## This file presents a 2D Savitzky-Golay filtering algorithm.
## This is a copy of the function found in the scipy cookbook:
##
##    https: // scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
##
################################################################################
import scipy.signal as signal
import numpy as np

# -----------------------------------------------------------------------------
def sgolay2d(z, window_size, order, derivative=None):
    """
    Apply 2D Savitzky-Golay filtering to the input array.
    This is a copy of the function found in the scipy cookbook:
    
    https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html

    Parameters:
     - z: numpy.ndarray
        Input array to be filtered.
     - window_size: int
        Size of the window used for filtering. Must be odd.
     - order: int
        Order of the polynomial expression used for filtering.
     - derivative: str, optional
        Type of derivative to compute. Can be 'col', 'row', 'both', or None (default).

    Returns:
     - filtered_array: numpy.ndarray or tuple of numpy.ndarray
        Filtered array or tuple of filtered arrays if derivative is 'both'.

    Raises:
     - ValueError: If window_size is even or if order is too high for the window size.
    """

    # Number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    # Check that the window size is odd and that the order is not too high
    if window_size % 2 == 0:
        raise ValueError('Savitsky-Golay window_size must be odd.')
    if window_size**2 < n_terms:
        raise ValueError(
            'Savitsky-Golay polynomial order is too high for the window size')

    # Get the half window size
    half_size = window_size // 2

    # Setup the exponents of the polynomial:
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k-n, n) for k in range(order+1) for n in range(k+1)]

    # Coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size**2, )

    # Build matrix of system of equations
    A = np.empty((window_size**2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx**exp[0]) * (dy**exp[1])

    # Pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros((new_shape))

    # Top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - \
        np.abs(np.flipud(z[1:half_size+1, :]) - band)

    # Bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + \
        np.abs(np.flipud(z[-half_size-1:-1, :]) - band)

    # Left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - \
        np.abs(np.fliplr(z[:, 1:half_size+1]) - band)

    # Right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + \
        np.abs(np.fliplr(z[:, -half_size-1:-1]) - band)

    # Central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # Top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - \
        np.abs(np.flipud(np.fliplr(z[1:half_size+1, 1:half_size+1])) - band)

    # Bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + \
        np.abs(
            np.flipud(np.fliplr(z[-half_size-1:-1, -half_size-1:-1])) - band)

    # Top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - \
        np.abs(np.flipud(Z[half_size+1:2*half_size+1, -half_size:]) - band)

    # Bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - \
        np.abs(np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band)

    # Solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return signal.fftconvolve(Z, -r, mode='valid'), signal.fftconvolve(Z, -c, mode='valid')

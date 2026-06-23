################################################################################
# OPTIMIZED 2D INTERPOLATOR FOR SUN-DIC
#
# This code is a highly specialized and optimized 2D fork based on the original 
# "fast_interp" library written by David Stein (https://github.com/dbstein/fast_interp).
# 
# Original code is licensed under the Apache License, Version 2.0.
# Copyright 2019 David Stein
#
# Modifications for SUN-DIC:
# - Stripped 1D and 3D implementations to strictly focus on 2D planar interpolation.
# - Replaced dynamic RAM allocation (NumPy arrays) inside loops with CPU-register tuples.
# - Replaced edge padding arrays with mathematical clamping for memory safety.
# - Applied aggressive Numba JIT compilation flags (cache=True, fastmath=True).
# - Simplified the Python class structure into a lightweight wrapper.
################################################################################

# Import libraries required by the code
import numpy as np
import numba

# --------------------------------------------------------------------------------------------
@numba.njit(cache=True, fastmath=True, parallel=True)
def _eval_interp2d_k3_(img, y_pts, x_pts, out):
    """
    Evaluate the 3rd order (bicubic) Taylor series interpolation for a given 
    image at specified sub-pixel coordinates.

    Parameters:
    - img (ndarray): The reference image array.
    - y_pts (ndarray): Flattened array of y-coordinates.
    - x_pts (ndarray): Flattened array of x-coordinates.
    - out (ndarray): Pre-allocated 1D output array to store interpolated intensities.
    """
    ny, nx = img.shape
    m = y_pts.shape[0]
    
    for mi in numba.prange(m):
        x_val = x_pts[mi]
        y_val = y_pts[mi]
        
        # Fast mathematical clamping to ensure coordinates stay within image bounds
        if x_val < 1.0: x_val = 1.0
        if x_val > nx - 2.0 - 1e-10: x_val = nx - 2.0 - 1e-10
        if y_val < 1.0: y_val = 1.0
        if y_val > ny - 2.0 - 1e-10: y_val = ny - 2.0 - 1e-10
        
        ix = int(x_val)
        iy = int(y_val)
        
        ratx = x_val - (ix + 0.5)
        raty = y_val - (iy + 0.5)
        
        # Unrolled Taylor expansion coefficients (Stored in CPU registers, bypassing RAM)
        asx0 = -1/16 + ratx*( 1/24 + ratx*( 1/4 - ratx/6))
        asx1 =  9/16 + ratx*( -9/8 + ratx*(-1/4 + ratx/2))
        asx2 =  9/16 + ratx*(  9/8 + ratx*(-1/4 - ratx/2))
        asx3 = -1/16 + ratx*(-1/24 + ratx*( 1/4 + ratx/6))
        
        asy0 = -1/16 + raty*( 1/24 + raty*( 1/4 - raty/6))
        asy1 =  9/16 + raty*( -9/8 + raty*(-1/4 + raty/2))
        asy2 =  9/16 + raty*(  9/8 + raty*(-1/4 - raty/2))
        asy3 = -1/16 + raty*(-1/24 + raty*( 1/4 + raty/6))
        
        ix_b = ix - 1
        iy_b = iy - 1
        
        val = 0.0
        asx = (asx0, asx1, asx2, asx3)
        asy = (asy0, asy1, asy2, asy3)
        
        # Local unrolled convolution
        for j in range(4):
            for i in range(4):
                val += img[iy_b + j, ix_b + i] * asx[i] * asy[j]
                
        out[mi] = val

# --------------------------------------------------------------------------------------------
@numba.njit(cache=True, fastmath=True, parallel=True)
def _eval_interp2d_k5_(img, y_pts, x_pts, out):
    """
    Evaluate the 5th order (biquintic) Taylor series interpolation for a given 
    image at specified sub-pixel coordinates.

    Parameters:
    - img (ndarray): The reference image array.
    - y_pts (ndarray): Flattened array of y-coordinates.
    - x_pts (ndarray): Flattened array of x-coordinates.
    - out (ndarray): Pre-allocated 1D output array to store interpolated intensities.
    """
    ny, nx = img.shape
    m = y_pts.shape[0]
    
    for mi in numba.prange(m):
        x_val = x_pts[mi]
        y_val = y_pts[mi]
        
        # Fast mathematical clamping adapted for order 5 bounds
        if x_val < 2.0: x_val = 2.0
        if x_val > nx - 3.0 - 1e-10: x_val = nx - 3.0 - 1e-10
        if y_val < 2.0: y_val = 2.0
        if y_val > ny - 3.0 - 1e-10: y_val = ny - 3.0 - 1e-10
        
        ix = int(x_val)
        iy = int(y_val)
        
        ratx = x_val - (ix + 0.5)
        raty = y_val - (iy + 0.5)
        
        # Unrolled Taylor expansion coefficients (Stored in CPU registers, bypassing RAM)
        asx0 =   3/256 + ratx*(   -9/1920 + ratx*( -5/48/2 + ratx*(  1/8/6 + ratx*( 1/2/24 -  1/8/120*ratx))))
        asx1 = -25/256 + ratx*(  125/1920 + ratx*( 39/48/2 + ratx*(-13/8/6 + ratx*(-3/2/24 +  5/8/120*ratx))))
        asx2 = 150/256 + ratx*(-2250/1920 + ratx*(-34/48/2 + ratx*( 34/8/6 + ratx*( 2/2/24 - 10/8/120*ratx))))
        asx3 = 150/256 + ratx*( 2250/1920 + ratx*(-34/48/2 + ratx*(-34/8/6 + ratx*( 2/2/24 + 10/8/120*ratx))))
        asx4 = -25/256 + ratx*( -125/1920 + ratx*( 39/48/2 + ratx*( 13/8/6 + ratx*(-3/2/24 -  5/8/120*ratx))))
        asx5 =   3/256 + ratx*(    9/1920 + ratx*( -5/48/2 + ratx*( -1/8/6 + ratx*( 1/2/24 +  1/8/120*ratx))))
        
        asy0 =   3/256 + raty*(   -9/1920 + raty*( -5/48/2 + raty*(  1/8/6 + raty*( 1/2/24 -  1/8/120*raty))))
        asy1 = -25/256 + raty*(  125/1920 + raty*( 39/48/2 + raty*(-13/8/6 + raty*(-3/2/24 +  5/8/120*raty))))
        asy2 = 150/256 + raty*(-2250/1920 + raty*(-34/48/2 + raty*( 34/8/6 + raty*( 2/2/24 - 10/8/120*raty))))
        asy3 = 150/256 + raty*( 2250/1920 + raty*(-34/48/2 + raty*(-34/8/6 + raty*( 2/2/24 + 10/8/120*raty))))
        asy4 = -25/256 + raty*( -125/1920 + raty*( 39/48/2 + raty*( 13/8/6 + raty*(-3/2/24 -  5/8/120*raty))))
        asy5 =   3/256 + raty*(    9/1920 + raty*( -5/48/2 + raty*( -1/8/6 + raty*( 1/2/24 +  1/8/120*raty))))
        
        ix_b = ix - 2
        iy_b = iy - 2
        
        val = 0.0
        asx = (asx0, asx1, asx2, asx3, asx4, asx5)
        asy = (asy0, asy1, asy2, asy3, asy4, asy5)
        
        # Local unrolled convolution
        for j in range(6):
            for i in range(6):
                val += img[iy_b + j, ix_b + i] * asx[i] * asy[j]
                
        out[mi] = val

# --------------------------------------------------------------------------------------------
class OptimizedInterp2D:
    """
    A lightweight wrapper class to replace the original heavy interp2d class.
    It manages the RAM allocation once and dispatches execution to Numba compiled functions.
    """
    def __init__(self, image, order=3):
        self.image = image
        self.order = order

    def __call__(self, y_pts, x_pts):
        y_flat = y_pts.ravel()
        x_flat = x_pts.ravel()
        
        # Allocate output RAM only once before the Numba JIT loop
        out = np.empty(y_flat.shape[0], dtype=self.image.dtype)
        
        if self.order == 3:
            _eval_interp2d_k3_(self.image, y_flat, x_flat, out)
        else:
            _eval_interp2d_k5_(self.image, y_flat, x_flat, out)
            
        return out.reshape(y_pts.shape)
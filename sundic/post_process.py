################################################################################
# This file contains the post processing functions for the sun-dic analysis.
# The functions are used to process the results returned by the sun-dic Digital
# Image Correlation (DIC) analysis.
##
# Author: G Venter
# Date: 2024/06/05
################################################################################
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline, LinearNDInterpolator
import sundic.sundic as sdic
from sundic.util.savitsky_golay import sgolay2d

# --------------------------------------------------------------------------------------------
class Comp(Enum):
    """n
    Enumeration representing different components used for displacement and strain fields.

    Attributes:
        - X (int): The X component.
        - Y (int): The Y component.
        - MAG (int): The magnitude component.
        - SHEAR (int): The shear component.
        - VM(int): The Von Mises component.
    """
    # General
    X = 3
    Y = 4

    # Displacement specific
    MAG = 6

    # Strain specific
    SHEAR = 5
    VM = 6


# --------------------------------------------------------------------------------------------
def getDisplacements(settings, nRows, nCols, subSetPoints, coeffs,
                     smoothWindow=0, smoothOrder=2):
    """
    Calculate and return the displacements based on the subset points and coefficients.

    Parameters:
     - settings (dict): Dictionary of settings - same values used to run sundic.
     - nRows (int): Number of rows in the dic data.
     - nCols (int): Number of columns in the dic data.
     - subSetPoints (ndarray): Array of subset points from sundic.
     - coeffs (ndarray): Coefficients for calculating the displacements from sundic.
     - smoothWindow (int, optional): Size of the window sisze used for the Savitzky-Golay
        smoothing.  Must be an odd number and a value of 0 indicates no smoothing. 
        Default is 0.
     - smoothOrder (int, optional): Order of the Savitzky-Golay smoothing polynomial. 
        Default is 2.

    Returns:
     - ndarray: Array of displacements.  The columns are as follows:
            - Column 0: x coordinate of the subset point.
            - Column 1: y coordinate of the subset point.
            - Column 2: z coordinate - 0's for now.
            - Column 3: x displacement component.
            - Column 4: y displacement component.
            - Column 5: z displacement component - 0's for now.
            - Column 6: displacement magnitude.

    Raises:
     - ValueError: If an invalid shapeFns argument is provided.
    """
    # Setup a results array
    nSubSets = coeffs.shape[1]
    results = np.zeros((nSubSets, 7))

    # Store the x and y coordinates of the subset points in the 1st and 2nd
    # columns of the results array
    results[:, 0:2] = subSetPoints.T

    # Store the x displacement component
    results[:, Comp.X.value] = coeffs[0, :].T

    # Get the y displacement component based on the shape functions used
    if settings['ShapeFunctions'] == 'Affine':
        results[:, Comp.Y.value] = np.copy(coeffs[3, :].T)

    elif settings['ShapeFunctions'] == 'Quadratic':
        results[:, Comp.Y.value] = np.copy(coeffs[6, :].T)

    else:
        raise ValueError(
            'Invalid shapeFns argument.  Only Affine and Quadratic are supported.')

    # Calculate the displacement magnitude and store in the 5th column of the
    # results array
    results[:, Comp.MAG.value] = np.sqrt(results[:, Comp.X.value]**2 +
                                         results[:, Comp.Y.value]**2)

    # If smoothing is requested, apply Savitzky-Golay smoothing
    if smoothWindow > 0:
        results[:, Comp.X.value] = __smoothResults__(nRows, nCols, results,
                                                     Comp.X.value, smoothWindow=smoothWindow, smoothOrder=smoothOrder)
        results[:, Comp.Y.value] = __smoothResults__(nRows, nCols, results,
                                                     Comp.Y.value, smoothWindow=smoothWindow, smoothOrder=smoothOrder)
        results[:, Comp.MAG.value] = __smoothResults__(nRows, nCols, results,
                                                       Comp.MAG.value, smoothWindow=smoothWindow, smoothOrder=smoothOrder)

    return results


# --------------------------------------------------------------------------------------------
def getStrains(settings, nRows, nCols, subSetPoints, coeffs, smoothWindow=3, smoothOrder=2):
    """
    Calculate and return the strains based on the subset points and coefficients.  For now only 
    Engineering Strain is calculated.

    Parameters:
     - settings (dict): Dictionary of settings - same values used to run sundic.
     - nRows (int): Number of rows in the dic data.
     - nCols (int): Number of columns in the dic data.
     - subSetPoints (ndarray): Array of subset points from sundic.
     - coeffs (ndarray): Coefficients for calculating the displacements from sundic.
     - smoothWindow (int, optional): Size of the window sisze used for the Savitzky-Golay
        smoothing.  Must be an odd number larger than 0.  Default is 3.
     - smoothOrder (int, optional): Order of the Savitzky-Golay smoothing polynomial. 
        Default is 2.

    Returns:
     - ndarray: Array of displacements.  The columns are as follows:
            - Column 0: x coordinate of the subset point.
            - Column 1: y coordinate of the subset point.
            - Column 2: z coordinate - 0's for now.
            - Column 3: x strain component.
            - Column 4: y strain component.
            - Column 5: xy/shear strain component.
            - Column 6: Von Mises strain.

    Raises:
     - ValueError: If an invalid smoothFactor argument is provided.
    """

    # Make sure the smoothFactor is larger than zero
    if smoothWindow <= 0:
        raise ValueError('smoothWindow must be larger than zero.')

    # Setup a results array
    nSubSets = coeffs.shape[1]
    results = np.zeros((nSubSets, 7))

    # Get the displacements - no smoothing yet
    disp = getDisplacements(settings, nRows, nCols, subSetPoints, coeffs,
                            smoothWindow=0)

    # Store the x and y coordinates of the subset points in the 1st and 2nd
    # columns of the results array
    results[:, 0:2] = subSetPoints.T

    # Apply Savitzky-Golay smoothing with gradient calculation
    dudy, dudx = __smoothResults__(
        nRows, nCols, disp, Comp.X.value, smoothWindow=smoothWindow,
        smoothOrder=smoothOrder, derivative='both')
    dvdy, dvdx = __smoothResults__(
        nRows, nCols, disp, Comp.Y.value, smoothWindow=smoothWindow,
        smoothOrder=smoothOrder, derivative='both')

    # Store the strain components
    results[:, Comp.X.value] = dudx
    results[:, Comp.Y.value] = dvdy
    results[:, Comp.SHEAR.value] = 0.5 * (dudy + dvdx)
    results[:, Comp.VM.value] = np.sqrt(results[:, Comp.X.value]**2 +
                                        results[:, Comp.Y.value]**2 -
                                        results[:, Comp.X.value] *
                                        results[:, Comp.Y.value] +
                                        3 * results[:, Comp.SHEAR.value]**2)

    return results


# --------------------------------------------------------------------------------------------
def plotDispContour(settings, nRows, nCols, subSetPoints, coeffs, dispComp=Comp.MAG,
                    alpha=0.75, plotImage=True, showPlot=True, fileName='',
                    smoothWindow=0, smoothOrder=2, maxValue=None, minValue=None):
    """
    Plot the displacement contour based on the subset points and coefficients.

    Parameters:
        - settings (dict): Dictionary of settings - same values used to run sundic.
        - nRows (int): Number of rows in the dic data.
        - nCols (int): Number of columns in the dic data.
        - subSetPoints (ndarray): Array of subset points from sundic.
        - coeffs (ndarray): Coefficients for calculating the displacements from sundic.
        - dispComp (Comp, optional): Component of the displacement to plot. Default is Comp.MAG.
        - alpha (float, optional): Transparency of the contour plot. Default is 0.75.
        - plotImage (bool, optional): Flag to plot the image under the contour plot. Default is True.
        - showPlot (bool, optional): Flag to show the plot. Default is True.
        - fileName (str, optional): Name of the file to save the plot. Default is ''.
        - smoothWindow (int, optional): Size of the window sisze used for the Savitzky-Golay
          smoothing.  Must be an odd number and a value of 0 indicates no smoothing. 
          Default is 0.
        - smoothOrder (int, optional): Order of the Savitzky-Golay smoothing polynomial. 
          Default is 2.
        - maxValue (float, optional): Maximum value to plot.  Default is None.
        - minValue (float, optional): Minimum value to plot.  Default is None.

    Raises:
        - ValueError: If an invalid dispComp argument is provided.
    """

    # Get the displacement results
    results = getDisplacements(settings, nRows, nCols, subSetPoints, coeffs,
                               smoothWindow, smoothOrder)

    # Setup the plot arrays
    X = results[:, 0].reshape(nCols, nRows)
    Y = results[:, 1].reshape(nCols, nRows)
    if dispComp == Comp.MAG:
        Z = results[:, Comp.MAG.value].reshape(nCols, nRows)
    elif dispComp == Comp.X:
        Z = results[:, Comp.X.value].reshape(nCols, nRows)
    elif dispComp == Comp.Y:
        Z = results[:, Comp.Y.value].reshape(nCols, nRows)
    else:
        raise ValueError('Invalid dispComp argument - use the Comp object.')

    # Apply maximum and minimum values if provided
    if maxValue:
        Z[Z > maxValue] = maxValue
    if minValue:
        Z[Z < minValue] = minValue

    # Read the image to plot on and plot
    if plotImage:
        imgSet = sdic.getImageList(settings['ImageFolder'])
        img = cv2.imread(imgSet[0], cv2.IMREAD_GRAYSCALE)
        plt.imshow(img, zorder=1, cmap='gray', vmin=0, vmax=255)

    # Setup the contour plot and plot on top of the image
    plt.contourf(X, Y, Z, alpha=alpha, zorder=2)
    cmap = plt.get_cmap('jet')
    plt.set_cmap(cmap)
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.colorbar()

    # Show and or save the plot
    if showPlot:
        plt.show()
    if fileName:
        plt.savefig(fileName)

    return


# --------------------------------------------------------------------------------------------
def plotStrainContour(settings, nRows, nCols, subSetPoints, coeffs, strainComp=Comp.X,
                      alpha=0.75, plotImage=True, showPlot=True, fileName='',
                      smoothWindow=3, smoothOrder=2, maxValue=None, minValue=None):
    """
    Plot the displacement contour based on the subset points and coefficients.

    Parameters:
        - settings (dict): Dictionary of settings - same values used to run sundic.
        - nRows (int): Number of rows in the dic data.
        - nCols (int): Number of columns in the dic data.
        - subSetPoints (ndarray): Array of subset points from sundic.
        - coeffs (ndarray): Coefficients for calculating the displacements from sundic.
        - dispComp (Comp, optional): Component of the displacement to plot. Default is Comp.MAG.
        - alpha (float, optional): Transparency of the contour plot. Default is 0.75.
        - plotImage (bool, optional): Flag to plot the image under the contour plot. Default is True.
        - showPlot (bool, optional): Flag to show the plot. Default is True.
        - fileName (str, optional): Name of the file to save the plot. Default is ''.
        - smoothWindow (int, optional): Size of the window sisze used for the Savitzky-Golay
          smoothing.  Must be an odd number larger than zero.  Default is 3.
        - smoothOrder (int, optional): Order of the Savitzky-Golay smoothing polynomial. 
          Default is 2.
        - maxValue (float, optional): Maximum value to plot.  Default is None.
        - minValue (float, optional): Minimum value to plot.  Default is None.

    Raises:
        - ValueError: If an invalid dispComp argument is provided.
    """

    # Get the displacement results
    results = getStrains(settings, nRows, nCols, subSetPoints, coeffs,
                         smoothWindow, smoothOrder)

    # Setup the plot arrays
    X = results[:, 0].reshape(nCols, nRows)
    Y = results[:, 1].reshape(nCols, nRows)
    if strainComp == Comp.SHEAR:
        Z = results[:, Comp.SHEAR.value].reshape(nCols, nRows)
    elif strainComp == Comp.X:
        Z = results[:, Comp.X.value].reshape(nCols, nRows)
    elif strainComp == Comp.Y:
        Z = results[:, Comp.Y.value].reshape(nCols, nRows)
    elif strainComp == Comp.VM:
        Z = results[:, Comp.VM.value].reshape(nCols, nRows)
    else:
        raise ValueError('Invalid strainComp argument - use the Comp object.')

    # Apply maximum and minimum values if provided
    if maxValue:
        Z[Z > maxValue] = maxValue
    if minValue:
        Z[Z < minValue] = minValue

    # Read the image to plot on and plot
    if plotImage:
        imgSet = sdic.getImageList(settings['ImageFolder'])
        img = cv2.imread(imgSet[0], cv2.IMREAD_GRAYSCALE)
        plt.imshow(img, zorder=1, cmap='gray', vmin=0, vmax=255)

    # Setup the contour plot and plot on top of the image
    plt.contourf(X, Y, Z, alpha=alpha, zorder=2)
    cmap = plt.get_cmap('jet')
    plt.set_cmap(cmap)
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.colorbar()

    # Show and or save the plot
    if showPlot:
        plt.show()
    if fileName:
        plt.savefig(fileName)

    return


# --------------------------------------------------------------------------------------------
def plotDispCutLine(settings, nRows, nCols, subSetPoints, coeffs, dispComp=Comp.X,
                    cutComp=Comp.Y, cutValues=[0], gridLines=True, showPlot=True,
                    fileName='', smoothWindow=0, smoothOrder=2, interpolate=False):
    """
    Plot a displacement cut line based on the subset points and coefficients.  The cut line
    is shown for the specified displacement component in specified direction.

    Parameters:
        - settings (dict): Dictionary of settings - same values used to run sundic.
        - nRows (int): Number of rows in the dic data.
        - nCols (int): Number of columns in the dic data.
        - subSetPoints (ndarray): Array of subset points from sundic.
        - coeffs (ndarray): Coefficients for calculating the displacements from sundic.
        - dispComp (Comp, optional): Component of the displacement to plot. Default is Comp.X.
        - cutComp (Comp, optional): Component of the cut line. Default is Comp.Y.
        - cutValues (list, optional): List of values to plot the cut line at. Default is [0].
        - gridLines (bool, optional): Flag to plot grid lines. Default is True.
        - showPlot (bool, optional): Flag to show the plot. Default is True.
        - fileName (str, optional): Name of the file to save the plot. Default is ''.
        - smoothWindow (int, optional): Size of the window sisze used for the Savitzky-Golay
          smoothing.  Must be an odd number and a value of 0 indicates no smoothing. 
          Default is 0.
        - smoothOrder (int, optional): Order of the Savitzky-Golay smoothing polynomial.
          Default is 2.
        - interpolate (bool, optional): Flag to interpolate the cut line. Default is False
          in which case the nearest neighbor is used.

    Raises:
        - ValueError: If an invalid dispComp or cutComp argument is provided.
    """

    # Get the displacement results
    results = getDisplacements(settings, nRows, nCols, subSetPoints, coeffs,
                               smoothWindow, smoothOrder)

    # Setup the plot arrays
    X = results[:, 0].reshape(nCols, nRows)
    X = X[:, 0]
    Y = results[:, 1].reshape(nCols, nRows)
    Y = Y[0, :]

    # Setup the y label based on the requested component
    ylabel = ''
    if dispComp == Comp.MAG:
        ylabel = 'Displacement Magnitude (pixels)'
    elif dispComp == Comp.X:
        ylabel = 'Displacement X (pixels)'
    elif dispComp == Comp.Y:
        ylabel = 'Displacement Y (pixels)'
    else:
        raise ValueError('Invalid dispComp argument - use the Comp object.')

    # Create the cutline plot
    plt = __createCutLineGraph__(nCols, nRows, results[:, 0], results[:, 1],
                                 results[:, dispComp.value], cutValues, cutComp,
                                 ylabel, interpolate)

    # Show gridlines if requested
    if gridLines:
        plt.grid()

    # Show and or save the plot
    if showPlot:
        plt.show()
    if fileName:
        plt.savefig(fileName)

    return


# --------------------------------------------------------------------------------------------
def plotStrainCutLine(settings, nRows, nCols, subSetPoints, coeffs, strainComp=Comp.X,
                      cutComp=Comp.Y, cutValues=[0],
                      gridLines=True, showPlot=True,
                      fileName='', smoothWindow=9, smoothOrder=2, interpolate=False):
    """
    Plot a strain cut line based on the subset points and coefficients.  The cut line
    is shown for the specified strain component in the specified direction.

    Parameters:
        - settings (dict): Dictionary of settings - same values used to run sundic.
        - nRows (int): Number of rows in the dic data.
        - nCols (int): Number of columns in the dic data.
        - subSetPoints (ndarray): Array of subset points from sundic.
        - coeffs (ndarray): Coefficients for calculating the displacements from sundic.
        - strainComp (Comp, optional): Component of the displacement to plot. Default is Comp.X.
        - cutComp (Comp, optional): Component of the cut line. Default is Comp.Y.
        - cutValues (list, optional): List of values to plot the cut line at. Default is [0].
        - gridLines (bool, optional): Flag to plot grid lines. Default is True.
        - showPlot (bool, optional): Flag to show the plot. Default is True.
        - fileName (str, optional): Name of the file to save the plot. Default is ''.
        - smoothWindow (int, optional): Size of the window sisze used for the Savitzky-Golay
          smoothing.  Must be an odd number and a value of 0 indicates no smoothing. 
          Default is 9.
        - smoothOrder (int, optional): Order of the Savitzky-Golay smoothing polynomial.
          Default is 2.
        - interpolate (bool, optional): Flag to interpolate the cut line. Default is False.

    Raises:
        - ValueError: If an invalid dispComp or cutComp argument is provided.
    """

    # Get the displacement results
    results = getStrains(settings, nRows, nCols, subSetPoints, coeffs,
                         smoothWindow, smoothOrder)

    # Setup the plot arrays
    X = results[:, 0].reshape(nCols, nRows)
    X = X[:, 0]
    Y = results[:, 1].reshape(nCols, nRows)
    Y = Y[0, :]
    ylabel = ''
    if strainComp == Comp.SHEAR:
        ylabel = 'Strain (XY component)'
        Z = results[:, Comp.SHEAR.value].reshape(nCols, nRows)
    elif strainComp == Comp.X:
        ylabel = 'Strain (X component)'
        Z = results[:, Comp.X.value].reshape(nCols, nRows)
    elif strainComp == Comp.Y:
        ylabel = 'Strain (Y component)'
        Z = results[:, Comp.Y.value].reshape(nCols, nRows)
    elif strainComp == Comp.VM:
        ylabel = 'Strain (Von Mises)'
        Z = results[:, Comp.VM.value].reshape(nCols, nRows)
    else:
        raise ValueError('Invalid strainComp argument - use the Comp object.')

    # Create the cutline plot
    plt = __createCutLineGraph__(nCols, nRows, results[:, 0], results[:, 1],
                                 results[:, strainComp.value], cutValues, cutComp,
                                 ylabel, interpolate)

    # Show gridlines if requested
    if gridLines:
        plt.grid()

    # Show and or save the plot
    if showPlot:
        plt.show()
    if fileName:
        plt.savefig(fileName)

    return


# --------------------------------------------------------------------------------------------
def __smoothResults__(nRows, nCols, results, comp, smoothWindow=3, smoothOrder=2,
                      derivative='none'):
    """
    Smooths the results of a computation over a grid using Savitzky-Golay smoothing.

    Parameters:
        - nRows (int): The number of rows in the grid.
        - nCols (int): The number of columns in the grid.
        - results (ndarray): The results matrix from the DIC values.
        - comp (int): The component of the results to smooth.
        - smoothWindow (int, optional): The size of the smoothing window. Defaults to 3.
        - smoothOrder (int, optional): The order of the smoothing. Defaults to 2.
        - derivative (str, optional): The type of derivative to compute. Defaults to 'none'.

    Returns:
        - ndarray or tuple: The smoothed results or the derivatives, 
          depending on the value of `derivative`.

    Raises:
        ValueError: If `smoothWindow` is not an odd number.
        ValueError: If `derivative` argument is invalid.
    """
    # Make sure the smoothWindow is odd and raise an exception if not
    if smoothWindow % 2 == 0:
        raise ValueError('smoothWindow must be an odd number.')

    # Get the result to smooth
    smoothRslt = results[:, comp]

    # Create a mask for all the non-nan values - will use later on to make the interpolated
    # values NaN again
    mask = ~np.isnan(smoothRslt)

    # Drop all NaN values and fill with nearest neighbor interpolation - only do this when
    # there are NaN values
    smoothRslt = __fillMissingData__(results[:, 0], results[:, 1], smoothRslt)

    # Apply Savitzky-Golay smoothing and reset the NaN values to indicate points not found
    if derivative == 'none':
        smoothRslt = sgolay2d(smoothRslt.reshape(
            nCols, nRows), smoothWindow, smoothOrder)
        smoothRslt[~mask.reshape(nCols, nRows)] = np.nan
        smoothRslt = smoothRslt.reshape(-1, order='C')

        return smoothRslt

    # If we asked for the derivatives
    elif derivative == 'both':
        drdc, drdr = sgolay2d(smoothRslt.reshape(
            nCols, nRows), smoothWindow, smoothOrder, derivative='both')
        drdc[~mask.reshape(nCols, nRows)] = np.nan
        drdr[~mask.reshape(nCols, nRows)] = np.nan
        drdc = drdc.reshape(-1, order='C')
        drdr = drdr.reshape(-1, order='C')

        return drdc, drdr

    # Else throw an exception
    else:
        raise ValueError(
            'Invalid derivative argument - only none or both are supported.')


# --------------------------------------------------------------------------------------------
def __fillMissingData__(dataX, dataY, dataVal):
    """
    Fill missing data values (specificall NaN's) using linear interpolation.

    Parameters:
      - dataX (numpy.ndarray): Array of x-coordinates.
      - dataY (numpy.ndarray): Array of y-coordinates.
      - dataVal (numpy.ndarray): Array of data values.

    Returns:
      - numpy.ndarray: Array of data values with missing values filled using 
        linear interpolation.
    """

    # Check if there are NaN values to interpolate
    if np.isnan(dataVal).any():

        # Get a mask for the values that are not NaN
        mask = ~np.isnan(dataVal)

        # Setup the linear interpolator
        interp = LinearNDInterpolator(
            list(zip(dataX[mask], dataY[mask])), dataVal[mask])

        # Interpoloate all nan values
        dataVal[~mask] = interp(dataX[~mask], dataY[~mask])

    return dataVal


# --------------------------------------------------------------------------------------------
def __createCutLineGraph__(nCols, nRows, dataX, dataY, dataZ, cutValues, cutComp,
                           ylabel, interpolate):
    """
    Create a cut line graph based on the given data.  Used for both displacement and
    strain plots

    Parameters:
        - nCols (int): The number of columns in the data.
        - nRows (int): The number of rows in the data.
        - dataX (ndarray): The X-coordinate data.
        - dataY (ndarray): The Y-coordinate data.
        - dataZ (ndarray): The values data.
        - cutValues (list): The values at which to make the cut lines.
        - cutComp (Comp): The component along which to make the cut lines (X or Y).
        - ylabel (str): The label for the Y-axis.
        - interpolate (bool): Whether to interpolate the data or use nearest values.

    Returns:
        - plt: The matplotlib plot object.

    Raises:
        None

    """

    # Process the raw data arrays
    X = dataX.reshape(nCols, nRows)
    X = X[:, 0]
    Y = dataY.reshape(nCols, nRows)
    Y = Y[0, :]

    # Setup the line styles and colours
    fig, ax = plt.subplots()
    colormap = plt.cm.hsv
    lsmap = ["-", ":", "--", "-."]
    ax.set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 1, len(cutValues))],
                      ls=np.resize(lsmap, len(cutValues)))

   # Setup the data to plot - first the interpolation case
    if interpolate:

        # Fill missing data and setup the interpolator
        Z = __fillMissingData__(dataX, dataY, dataZ)
        Z = Z.reshape(nCols, nRows)
        rbs = RectBivariateSpline(X, Y, Z, kx=3, ky=3)

        # Setup the x and y data and create the plot depending on the cutComp
        if cutComp == Comp.X:
            # Setup the data
            xlabel = 'y (pixels)'
            x = np.linspace(np.min(Y), np.max(Y), 101)
            y = np.dot(np.ones((x.shape[0], 1)), np.array(
                cutValues).reshape(1, len(cutValues)))

            # Make the plots based on the interpolation
            for col in range(0, y.shape[1]):
                z = rbs.ev(y[:, col], x)
                label = "x={0:d} px".format(cutValues[col])
                plt.plot(x, z, label=label)

        elif cutComp == Comp.Y:
            # Setup the data
            xlabel = 'x (pixels)'
            x = np.linspace(np.min(X), np.max(X), 101)
            y = np.dot(np.ones((x.shape[0], 1)), np.array(
                cutValues).reshape(1, len(cutValues)))

            # Make the plots based on the interpolation
            for col in range(0, y.shape[1]):
                z = rbs.ev(x, y[:, col])
                label = "y={0:d} px".format(cutValues[col])
                plt.plot(x, z, label=label)

        # Add the legend and the x, y labels
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    # No interpolation - get nearest values
    else:
        # Reshape the value data
        Z = dataZ.reshape(nCols, nRows)

        # Get nearest neighbor indices for cutlines
        indices = np.zeros_like(cutValues)
        if cutComp == Comp.X:
            # Setup the data
            xlabel = 'y (pixels)'
            for idx, val in enumerate(cutValues):
                indices[idx] = np.abs(X - val).argmin()
            x = Y
            y = Z[indices, :]
            for i in range(0, len(cutValues)):
                label = "x={0:d} px".format(int(X[indices[i]]))
                plt.plot(x, y[i, :], label=label)
            plt.legend()

        elif cutComp == Comp.Y:
            # Setup the data
            xlabel = 'x (pixels)'
            for idx, val in enumerate(cutValues):
                indices[idx] = np.abs(Y - val).argmin()
            x = X
            y = Z[:, indices]
            for i in range(0, len(cutValues)):
                label = "y={0:d} px".format(int(X[indices[i]]))
                plt.plot(x, y[:, i], label=label)
            plt.legend()

        # Display the x and y labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    return plt

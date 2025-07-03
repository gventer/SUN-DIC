################################################################################
# This file contains the post processing functions for the sun-dic analysis.
# The functions are used to process the results returned by the sun-dic Digital
# Image Correlation (DIC) analysis.
##
# Author: G Venter
# Date: 2024/06/05
################################################################################
from enum import IntEnum
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline, NearestNDInterpolator
import sundic.sundic as sdic
from sundic.util.savitsky_golay import sgolay2d
import sundic.util.datafile as dataFile
import sundic.settings as sdset

# --------------------------------------------------------------------------------------------


class DispComp(IntEnum):
    """
    Enumeration representing different components used for displacementfields.

    Attributes:
        - X_DISP (int): The X component.
        - Y_DISP (int): The Y component.
        - Z_DISP (int): The Z component.
        - DISP_MAG (int): The magnitude component.
    """
    X_DISP = (3, 'X Displacement')
    Y_DISP = (4, 'Y Displacement')
    Z_DISP = (5, 'Z Displacement')
    DISP_MAG = (6, 'Displacement Magnitude')

    # Add a display name to the enumeration
    def __new__(cls, value, display_name=None):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.display_name = display_name
        return obj


class StrainComp(IntEnum):
    """
    Enumeration representing different components used for strain fields.

    Attributes:
        - X_STRAIN (int): The X component.
        - Y_STRAIN (int): The Y component.
        - SHEAR_STRAIN (int): The shear component.
        - VM_STRAIN (int): The Von Mises component.
    """
    X_STRAIN = (3, 'X Strain')
    Y_STRAIN = (4, 'Y Strain')
    SHEAR_STRAIN = (5, 'Shear Strain')
    VM_STRAIN = (6, 'Von Mises Strain')

    # Add a display name to the enumeration
    def __new__(cls, value, display_name=None):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.display_name = display_name
        return obj


class CompID(IntEnum):
    """
    Enumeration representing different components used for displacement and strain fields.

    Attributes:
        - XCoordID (int): The X coordinate.
        - YCoordID (int): The Y coordinate.
        - XDispID (int): The X displacement.
    """
    XCoordID = (0, 'X Coordinate')
    YCoordID = (1, 'Y Coordinate')
    SSSizeID = 2   # The subset size
    ShapeFnID = 3   # The shape function - 0 = affine, 1 = quadratic
    CZNSSDID = 4   # The CZNSSD value for the subset
    XDispID = 5   # The x-displacement of the subset point - start of x model coefficients
    YDispID = 11  # The y-displacement of the subset point - start of y model coefficients

    # Add a display name to the enumeration
    def __new__(cls, value, display_name=None):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.display_name = display_name
        return obj


# --------------------------------------------------------------------------------------------
def getDisplacements(resultsFile, imgPair, smoothWindow=0, smoothOrder=2):
    """
    Calculate and return the displacements based on the results file created by 
    SUN-DIC.

    Parameters:
     - resultsFile (string): Results file from sundic.
     - imgPair (int): Zero based image pair to post-process for displacement values.
                    Use -1 for final/last pair.
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
    - nRows: int.  The number of rows of subsets.
    - nCols: int.  The number of columns of subsets.

    Raises:
     - ValueError: If an invalid shapeFns argument is provided.
    """
    # Load the results file and unpack the data
    inFile = dataFile.DataFile.openReader(resultsFile)

    # Ingore the heading
    _, _, setDict = inFile.readHeading()
    settings = sdset.Settings.fromMsgPackDict(setDict)

    # Get the data
    subSetPnts = inFile.readSubSetData(imgPair)

    # Close the file
    inFile.close()

    # Setup a results array
    nSubSets = subSetPnts.shape[0] * subSetPnts.shape[1]
    results = np.zeros((nSubSets, 7))

    # Store the x and y coordinates of the subset points in the 1st and 2nd
    # columns of the results array
    results[:, CompID.XCoordID] = subSetPnts[:, :,
                                             CompID.XCoordID].reshape(nSubSets, order='F')
    results[:, CompID.YCoordID] = subSetPnts[:, :,
                                             CompID.YCoordID].reshape(nSubSets, order='F')

    # Store the x displacement component
    results[:, DispComp.X_DISP] = subSetPnts[:, :,
                                             CompID.XDispID].reshape(nSubSets, order='F')

    # Get the y displacement component based on the shape functions used
    results[:, DispComp.Y_DISP] = subSetPnts[:, :,
                                             CompID.YDispID].reshape(nSubSets, order='F')

    # Calculate the displacement magnitude and store in the correct column of the
    # results array
    results[:, DispComp.DISP_MAG] = np.sqrt(results[:, DispComp.X_DISP]**2 +
                                            results[:, DispComp.Y_DISP]**2)

    # If smoothing is requested, apply Savitzky-Golay smoothing
    nRows = subSetPnts.shape[0]
    nCols = subSetPnts.shape[1]
    if smoothWindow > 0:
        results[:, DispComp.X_DISP] = _smoothResults_(nRows, nCols, results,
                                                      DispComp.X_DISP, smoothWindow=smoothWindow,
                                                      smoothOrder=smoothOrder)
        results[:, DispComp.Y_DISP] = _smoothResults_(nRows, nCols, results,
                                                      DispComp.Y_DISP, smoothWindow=smoothWindow,
                                                      smoothOrder=smoothOrder)
        results[:, DispComp.DISP_MAG] = _smoothResults_(nRows, nCols, results,
                                                        DispComp.DISP_MAG, smoothWindow=smoothWindow,
                                                        smoothOrder=smoothOrder)

    return results, nRows, nCols


# --------------------------------------------------------------------------------------------
def getCznssd(resultsFile, imgPair):
    """
    Calculate and return the Cznssd values for each subset based on the results file created by 
    SUN-DIC.

    Parameters:
     - resultsFile (string): Results file from sundic.
     - imgPair (int): Zero based image pair to post-process for displacement values.
                    Use -1 for final/last pair.

    Returns:
     - ndarray: Array of Cznssd values.  The columns are as follows:
            - Column 0: x coordinate of the subset point.
            - Column 1: y coordinate of the subset point.
            - Column 2: z coordinate - 0's for now.
            - Column 3: Cznssd values.
    - nRows: int.  The number of rows of subsets.
    - nCols: int.  The number of columns of subsets.
    """
    # Load the results file and unpack the data
    inFile = dataFile.DataFile.openReader(resultsFile)

    # Ingore the heading
    _, _, setDict = inFile.readHeading()
    settings = sdset.Settings.fromMsgPackDict(setDict)

    # Get the data
    subSetPnts = inFile.readSubSetData(imgPair)

    # Close the file
    inFile.close()

    # Setup a results array
    nRows = subSetPnts.shape[0]
    nCols = subSetPnts.shape[1]
    nSubSets = subSetPnts.shape[0] * subSetPnts.shape[1]
    results = np.zeros((nSubSets, 4))

    # Store the x and y coordinates of the subset points in the 1st and 2nd
    # columns of the results array
    results[:, CompID.XCoordID] = subSetPnts[:, :,
                                             CompID.XCoordID].reshape(nSubSets, order='F')
    results[:, CompID.YCoordID] = subSetPnts[:, :,
                                             CompID.YCoordID].reshape(nSubSets, order='F')

    # Store the Cznssd component
    results[:, -1] = subSetPnts[:, :,
                                CompID.CZNSSDID].reshape(nSubSets, order='F')
    results[:, -1][results[:, -1] == sdic.IntConst.CNZSSD_MAX] = np.nan

    return results, nRows, nCols


# --------------------------------------------------------------------------------------------
def getStrains(resultsFile, imgPair, smoothWindow=3, smoothOrder=2):
    """
    Calculate and return the strains based on the subset points and coefficients.  For now only 
    Engineering Strain is calculated.

    Parameters:
     - resultsFile (string): Results file from sundic.
     - imgPair (int): Zero based image pair to post-process for displacement values.
                    Use -1 for final/last pair.
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
    - nRows: int.  The number of rows of subsets.
    - nCols: int.  The number of columns of subsets.            

    Raises:
     - ValueError: If an invalid smoothFactor argument is provided.
    """

    # Make sure the smoothFactor is larger than zero
    if smoothWindow <= 0:
        raise ValueError('smoothWindow must be larger than zero.')

    # Get the displacements - no smoothing yet
    disp, nRows, nCols = getDisplacements(resultsFile, imgPair, smoothWindow=0)

    # Setup a results array
    results = np.zeros((nRows*nCols, 7))

    # Store the x and y coordinates of the subset points in the 1st and 2nd
    # columns of the results array
    results[:, CompID.XCoordID] = disp[:, CompID.XCoordID]
    results[:, CompID.YCoordID] = disp[:, CompID.YCoordID]

    # Apply Savitzky-Golay smoothing with gradient calculation
    dudy, dudx = _smoothResults_(
        nRows, nCols, disp, DispComp.X_DISP, smoothWindow=smoothWindow,
        smoothOrder=smoothOrder, derivative='both')
    dvdy, dvdx = _smoothResults_(
        nRows, nCols, disp, DispComp.Y_DISP, smoothWindow=smoothWindow,
        smoothOrder=smoothOrder, derivative='both')

    # Store the strain components
    results[:, StrainComp.X_STRAIN] = dudx
    results[:, StrainComp.Y_STRAIN] = dvdy
    results[:, StrainComp.SHEAR_STRAIN] = 0.5 * (dudy + dvdx)
    results[:, StrainComp.VM_STRAIN] = np.sqrt(results[:, StrainComp.X_STRAIN]**2 +
                                               results[:, StrainComp.Y_STRAIN]**2 -
                                               results[:, StrainComp.X_STRAIN] *
                                               results[:, StrainComp.Y_STRAIN] +
                                               3 * results[:, StrainComp.SHEAR_STRAIN]**2)

    return results, nRows, nCols


# --------------------------------------------------------------------------------------------
def plotDispContour(resultsFile, imgPair, dispComp=DispComp.DISP_MAG,
                    alpha=0.75, plotImage=True, showPlot=True, fileName='',
                    smoothWindow=0, smoothOrder=2, maxValue=None, minValue=None, return_fig=False):
    """
    Plot the displacement contour based on the subset points and coefficients.

    Parameters:
        - resultsFile (string): Results file from sundic.
        - imgPair (int): Zero based image pair to post-process for displacement values.
                    Use -1 for final/last pair.
        - dispComp (Comp, optional): Component of the displacement to plot. 
                    Default is DispComp.DISP_MAG.
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

    Returns: 
        - fig: The matplotlib plot object.

    Raises:
        - ValueError: If an invalid dispComp argument is provided.
    """

    # Get the displacement results
    results, nRows, nCols = getDisplacements(
        resultsFile, imgPair, smoothWindow, smoothOrder)

    # Setup the plot arrays
    X = results[:, CompID.XCoordID].reshape(nCols, nRows)
    Y = results[:, CompID.YCoordID].reshape(nCols, nRows)
    if dispComp == DispComp.DISP_MAG:
        Z = results[:, DispComp.DISP_MAG].reshape(nCols, nRows)
    elif dispComp == DispComp.X_DISP:
        Z = results[:, DispComp.X_DISP].reshape(nCols, nRows)
    elif dispComp == DispComp.Y_DISP:
        Z = results[:, DispComp.Y_DISP].reshape(nCols, nRows)
    else:
        raise ValueError('Invalid dispComp argument - use the Comp object.')

    # Apply maximum and minimum values if provided
    if maxValue:
        Z[Z > maxValue] = maxValue
    if minValue:
        Z[Z < minValue] = minValue

    # Get the settings object
    inFile = dataFile.DataFile.openReader(resultsFile)
    _, _, setDict = inFile.readHeading()
    settings = sdset.Settings.fromMsgPackDict(setDict)
    inFile.close()

    # Create figure object
    fig, ax = plt.subplots()

    # Read the image to plot on and plot
    if plotImage:
        imgSet = sdic._getImageList_(settings.ImageFolder)
        if imgPair == -1:
            imgPair = len(imgSet) - 1
        else:
            imgPair = imgPair + 1
        img = sdic.readImage(imgSet[imgPair], normalize8Bit=True)
        ax.imshow(img, zorder=1, cmap='gray', vmin=0, vmax=255)

    # Setup the contour plot and plot on top of the image
    contour = ax.contourf(X, Y, Z, alpha=alpha, zorder=2, cmap='jet')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    fig.colorbar(contour, ax=ax)

    # Show and or save the plot
    if showPlot:
        plt.show()
    if fileName:
        plt.savefig(fileName)

    if return_fig:
        return fig


# --------------------------------------------------------------------------------------------
def plotStrainContour(resultsFile, imgPair, strainComp=StrainComp.VM_STRAIN,
                      alpha=0.75, plotImage=True, showPlot=True, fileName='',
                      smoothWindow=3, smoothOrder=2, maxValue=None, minValue=None, return_fig=False):
    """
    Plot the displacement contour based on the subset points and coefficients.

    Parameters:
        - resultsFile (string): Results file from sundic.
        - imgPair (int): Zero based image pair to post-process for displacement values.
                    Use -1 for final/last pair.
        - strainComp (Comp, optional): Component of the strain to plot. 
                    Default is StrainComp.VM_STRAIN.
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
        - return_fig (bool, optional): Flag to return the figure object. Default is False.

    Returns: 
        - fig: The matplotlib plot object.

    Raises:
        - ValueError: If an invalid dispComp argument is provided.
    """

    # Get the strain results
    results, nRows, nCols = getStrains(
        resultsFile, imgPair, smoothWindow, smoothOrder)

    # Setup the plot arrays
    X = results[:, CompID.XCoordID].reshape(nCols, nRows)
    Y = results[:, CompID.YCoordID].reshape(nCols, nRows)

    if strainComp == StrainComp.SHEAR_STRAIN:
        Z = results[:, StrainComp.SHEAR_STRAIN].reshape(nCols, nRows)
    elif strainComp == StrainComp.X_STRAIN:
        Z = results[:, StrainComp.X_STRAIN].reshape(nCols, nRows)
    elif strainComp == StrainComp.Y_STRAIN:
        Z = results[:, StrainComp.Y_STRAIN].reshape(nCols, nRows)
    elif strainComp == StrainComp.VM_STRAIN:
        Z = results[:, StrainComp.VM_STRAIN].reshape(nCols, nRows)
    else:
        raise ValueError('Invalid strainComp argument - use the Comp object.')

    # Apply maximum and minimum values if provided
    if maxValue:
        Z[Z > maxValue] = maxValue
    if minValue:
        Z[Z < minValue] = minValue

    # Get the settings object
    inFile = dataFile.DataFile.openReader(resultsFile)
    _, _, setDict = inFile.readHeading()
    settings = sdset.Settings.fromMsgPackDict(setDict)
    inFile.close()

    # Create figure object
    fig, ax = plt.subplots()

    # Read the image to plot on and plot
    if plotImage:
        imgSet = sdic._getImageList_(settings.ImageFolder)
        if imgPair == -1:
            imgPair = len(imgSet) - 1
        else:
            imgPair = imgPair + 1
        img = sdic.readImage(imgSet[imgPair], normalize8Bit=True)
        ax.imshow(img, zorder=1, cmap='gray', vmin=0, vmax=255)

    # Setup the contour plot and plot on top of the image
    contour = ax.contourf(X, Y, Z, alpha=alpha, zorder=2, cmap='jet')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    fig.colorbar(contour, ax=ax)

    # Show and or save the plot
    if showPlot:
        plt.show()
    if fileName:
        plt.savefig(fileName)

    if return_fig:
        return fig


# --------------------------------------------------------------------------------------------
def plotZNCCContour(resultsFile, imgPair, alpha=0.75, plotImage=True, showPlot=True, fileName='',
                    maxValue=None, minValue=None, return_fig=False):
    """
    Plot the displacement contour based on the subset points and coefficients.

    Parameters:
        - resultsFile (string): Results file from sundic.
        - imgPair (int): Zero based image pair to post-process for displacement values.
                    Use -1 for final/last pair.
        - alpha (float, optional): Transparency of the contour plot. Default is 0.75.
        - plotImage (bool, optional): Flag to plot the image under the contour plot. Default is True.
        - showPlot (bool, optional): Flag to show the plot. Default is True.
        - fileName (str, optional): Name of the file to save the plot. Default is ''.
        - maxValue (float, optional): Maximum value to plot.  Default is None.
        - minValue (float, optional): Minimum value to plot.  Default is None.
        - return_fig (bool, optional): Flag to return the figure object. Default is False.

    Returns: 
        - fig: The matplotlib plot object.

    Raises:
        - ValueError: If an invalid dispComp argument is provided.
    """

    # Get the displacement results
    Cznssd, nRows, nCols = getCznssd(resultsFile, imgPair)

    # Setup the plot arrays
    X = Cznssd[:, CompID.XCoordID].reshape(nCols, nRows)
    Y = Cznssd[:, CompID.YCoordID].reshape(nCols, nRows)

    # Calculate the ZNCC form the stored Cznssd values
    Z = (1. - 0.5*Cznssd[:, -1]).reshape(nCols, nRows)

    # Apply maximum and minimum values if provided
    if maxValue:
        Z[Z > maxValue] = maxValue
    if minValue:
        Z[Z < minValue] = minValue

    # Get the settings object
    inFile = dataFile.DataFile.openReader(resultsFile)
    _, _, setDict = inFile.readHeading()
    settings = sdset.Settings.fromMsgPackDict(setDict)
    inFile.close()

    # Create figure object
    fig, ax = plt.subplots()

    # Read the image to plot on and plot
    if plotImage:
        imgSet = sdic._getImageList_(settings.ImageFolder)
        if imgPair == -1:
            imgPair = len(imgSet) - 1
        else:
            imgPair = imgPair + 1
        img = sdic.readImage(imgSet[imgPair], normalize8Bit=True)
        ax.imshow(img, zorder=1, cmap='gray', vmin=0, vmax=255)

    # Setup the contour plot and plot on top of the image
    contour = ax.contourf(X, Y, Z, alpha=alpha, zorder=2, cmap='jet')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    fig.colorbar(contour, ax=ax)

    # Show and or save the plot
    if showPlot:
        plt.show()
    if fileName:
        plt.savefig(fileName)

    if return_fig:
        return fig


# --------------------------------------------------------------------------------------------
def plotDispCutLine(resultsFile, imgPair, dispComp=DispComp.DISP_MAG, cutComp=CompID.YCoordID,
                    cutValues=[0], gridLines=True, showPlot=True, fileName='',
                    smoothWindow=0, smoothOrder=2, interpolate=False, return_fig=False):
    """
    Plot a displacement cut line based on the subset points and coefficients.  The cut line
    is shown for the specified displacement component in specified direction.

    Parameters:
        - resultsFile (string): Results file from sundic.
        - imgPair (int): Zero based image pair to post-process for displacement values.
                    Use -1 for final/last pair.
        - dispComp (Comp, optional): Component of the displacement to plot. 
            Default is DispComp.DISP_MAG.
        - cutComp (Comp, optional): Component of the cut line. Default is CompID.YCoordID.
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

    Returns: 
        - fig: The matplotlib plot object.

    Raises:
        - ValueError: If an invalid dispComp or cutComp argument is provided.
    """

    # Get the displacement results
    results, nRows, nCols = getDisplacements(
        resultsFile, imgPair, smoothWindow, smoothOrder)

    # Setup the plot arrays
    X = results[:, CompID.XCoordID].reshape(nCols, nRows)
    X = X[:, 0]
    Y = results[:, CompID.YCoordID].reshape(nCols, nRows)
    Y = Y[0, :]

    # Setup the y label based on the requested component
    ylabel = ''
    if dispComp == DispComp.DISP_MAG:
        ylabel = 'Displacement Magnitude (pixels)'
    elif dispComp == DispComp.X_DISP:
        ylabel = 'Displacement X (pixels)'
    elif dispComp == DispComp.Y_DISP:
        ylabel = 'Displacement Y (pixels)'
    else:
        raise ValueError('Invalid dispComp argument - use the Comp object.')

    # Create the cutline plot
    fig, ax = _createCutLineGraph_(nCols, nRows, results[:, CompID.XCoordID],
                                   results[:, CompID.YCoordID], results[:,
                                                                        dispComp.value],
                                   cutValues, cutComp, ylabel, interpolate)

    # Show gridlines if requested
    if gridLines:
        ax.grid()

    # Show and or save the plot
    if showPlot:
        plt.show()
    if fileName:
        plt.savefig(fileName)

    if return_fig:
        return fig


# --------------------------------------------------------------------------------------------
def plotStrainCutLine(resultsFile, imgPair, strainComp=StrainComp.VM_STRAIN,
                      cutComp=CompID.YCoordID, cutValues=[0],
                      gridLines=True, showPlot=True, fileName='',
                      smoothWindow=9, smoothOrder=2, interpolate=False, return_fig=False):
    """
    Plot a strain cut line based on the subset points and coefficients.  The cut line
    is shown for the specified strain component in the specified direction.

    Parameters:
        - resultsFile (string): Results file from sundic.
        - imgPair (int): Zero based image pair to post-process for displacement values.
                    Use -1 for final/last pair.
        - strainComp (Comp, optional): Component of the displacement to plot. 
            Default is StrainComp.VM_STRAIN
        - cutComp (Comp, optional): Component of the cut line. 
            Default is CompID.YCoordID
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

    Returns: 
        - fig: The matplotlib plot object.

    Raises:
        - ValueError: If an invalid dispComp or cutComp argument is provided.
    """

    # Get the displacement results
    results, nRows, nCols = getStrains(
        resultsFile, imgPair, smoothWindow, smoothOrder)

    # Setup the plot arrays
    X = results[:, CompID.XCoordID].reshape(nCols, nRows)
    X = X[:, 0]
    Y = results[:, CompID.YCoordID].reshape(nCols, nRows)
    Y = Y[0, :]
    ylabel = ''
    if strainComp == StrainComp.SHEAR_STRAIN:
        ylabel = 'Strain (XY component)'
        Z = results[:, StrainComp.SHEAR_STRAIN].reshape(nCols, nRows)
    elif strainComp == StrainComp.X_STRAIN:
        ylabel = 'Strain (X component)'
        Z = results[:, StrainComp.X_STRAIN].reshape(nCols, nRows)
    elif strainComp == StrainComp.Y_STRAIN:
        ylabel = 'Strain (Y component)'
        Z = results[:, StrainComp.Y_STRAIN].reshape(nCols, nRows)
    elif strainComp == StrainComp.VM_STRAIN:
        ylabel = 'Strain (Von Mises)'
        Z = results[:, StrainComp.VM_STRAIN].reshape(nCols, nRows)
    else:
        raise ValueError('Invalid strainComp argument - use the Comp object.')

    # Create the cutline plot
    fig, ax = _createCutLineGraph_(nCols, nRows, results[:, CompID.XCoordID],
                                   results[:, CompID.YCoordID], results[:,
                                                                        strainComp.value],
                                   cutValues, cutComp, ylabel, interpolate)

    # Show gridlines if requested
    if gridLines:
        ax.grid()

    # Show and or save the plot
    if showPlot:
        plt.show()
    if fileName:
        plt.savefig(fileName)

    if return_fig:
        return fig


# --------------------------------------------------------------------------------------------
def _smoothResults_(nRows, nCols, results, comp, smoothWindow=3, smoothOrder=2,
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
    smoothRslt = _fillMissingData_(results[:, CompID.XCoordID],
                                   results[:, CompID.YCoordID], smoothRslt)

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
def _fillMissingData_(dataX, dataY, dataVal):
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

        # Setup the nearest neighbor interpolator
        # interp = LinearNDInterpolator(
        interp = NearestNDInterpolator(
            list(zip(dataX[mask], dataY[mask])), dataVal[mask])

        # Interpoloate all nan values
        dataVal[~mask] = interp(dataX[~mask], dataY[~mask])

    return dataVal


# --------------------------------------------------------------------------------------------
def _createCutLineGraph_(nCols, nRows, dataX, dataY, dataZ, cutValues, cutComp,
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
        - fig: The matplotlib plot object.
        - ax: The matplotlib axis object.

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
        Z = _fillMissingData_(dataX, dataY, dataZ)
        Z = Z.reshape(nCols, nRows)
        rbs = RectBivariateSpline(X, Y, Z, kx=3, ky=3)

        # Setup the x and y data and create the plot depending on the cutComp
        if cutComp == CompID.XCoordID:
            # Setup the data
            xlabel = 'y (pixels)'
            x = np.linspace(np.min(Y), np.max(Y), 101)
            y = np.dot(np.ones((x.shape[0], 1)), np.array(
                cutValues).reshape(1, len(cutValues)))

            # Make the plots based on the interpolation
            for col in range(0, y.shape[1]):
                z = rbs.ev(y[:, col], x)
                label = "x={0:d} px".format(cutValues[col])
                ax.plot(x, z, label=label)

        elif cutComp == CompID.YCoordID:
            # Setup the data
            xlabel = 'x (pixels)'
            x = np.linspace(np.min(X), np.max(X), 101)
            y = np.dot(np.ones((x.shape[0], 1)), np.array(
                cutValues).reshape(1, len(cutValues)))

            # Make the plots based on the interpolation
            for col in range(0, y.shape[1]):
                z = rbs.ev(x, y[:, col])
                label = "y={0:d} px".format(cutValues[col])
                ax.plot(x, z, label=label)

        # Add the legend and the x, y labels
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # No interpolation - get nearest values
    else:
        # Reshape the value data
        Z = dataZ.reshape(nCols, nRows)

        # Get nearest neighbor indices for cutlines
        indices = np.zeros_like(cutValues)
        if cutComp == CompID.XCoordID:
            # Setup the data
            xlabel = 'y (pixels)'
            for idx, val in enumerate(cutValues):
                indices[idx] = np.abs(X - val).argmin()
            x = Y
            y = Z[indices, :]
            for i in range(0, len(cutValues)):
                label = "x={0:d} px".format(int(X[indices[i]]))
                ax.plot(x, y[i, :], label=label)
            ax.legend()

        elif cutComp == CompID.YCoordID:
            # Setup the data
            xlabel = 'x (pixels)'
            for idx, val in enumerate(cutValues):
                indices[idx] = np.abs(Y - val).argmin()
            x = X
            y = Z[:, indices]
            for i in range(0, len(cutValues)):
                label = "y={0:d} px".format(int(Y[indices[i]]))
                ax.plot(x, y[:, i], label=label)
            ax.legend()

        # Display the x and y labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    return fig, ax

# Importing required modules
import sundic.version as sdversion
from sundic.gui.widgets import *
from sundic.gui.validators import *
from sundic.gui.mainWindow import Ui_MainWindow
import webbrowser
import subprocess
import threading
import io
import os
import sys
import ray
import msgpack
import pandas as pd
import numpy as np
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import sip
from PyQt5 import *
from PyQt5 import QtWidgets

import matplotlib.pyplot
import natsort as ns
import sundic.settings as sdset
import sundic.sundic as sd
import sundic.util.datafile as df
import sundic.post_process as sdpp
import time
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar


import matplotlib
matplotlib.use('Qt5Agg')


# Default scale factor for image
SCALE_FACTOR = 1.25


class PhotoViewer(QGraphicsView):
    """
    A custom QGraphicsView for loading, displaying, and manipulating images to define areas of interest.
    Attributes:
        coordinatesChanged (pyqtSignal): Signal emitted when the coordinates change.
        rectDrawn (pyqtSignal): Signal emitted when a rectangle is drawn.
        _zoom (int): Current zoom level.
        _pinned (bool): Flag indicating if the zoom is pinned.
        _empty (bool): Flag indicating if the viewer is empty.
        _scene (QGraphicsScene): The graphics scene.
        _photo (QGraphicsPixmapItem): The pixmap item for the photo.
        clickedCounter (int): Counter for mouse clicks.
        areaOfInterestCoords (list): Coordinates of the area of interest.
        flag01 (bool): Flag indicating if the rectangle has been drawn by clicking.
        flag02 (bool): Flag indicating if the rectangle has been drawn by manual input.
        flag03 (bool): Flag indicating the need to remove the drawn rectangle in place of the manual rectangle.
        flag04 (bool): Flag indicating the need to remove the manual rectangle in place of the drawn rectangle.
    Methods:
        __init__(parent): Initializes the PhotoViewer.
        hasPhoto(): Checks if the viewer has a photo.
        resetView(scale=1): Resets the view to the initial state.
        setPhoto(pixmap=None): Sets the photo to be displayed.
        zoomLevel(): Returns the current zoom level.
        zoomPinned(): Checks if the zoom is pinned.
        setZoomPinned(enable): Sets the zoom pinned state.
        zoom(step): Zooms in or out by a given step.
        wheelEvent(event): Handles the mouse wheel event for zooming.
        resizeEvent(event): Handles the resize event.
        toggleDragMode(): Toggles the drag mode.
        updateCoordinates(pos=None): Updates the coordinates of the mouse position.
        leaveEvent(event): Handles the leave event.
        enterEvent(event): Handles the enter event.
        mousePressEvent(event, pos=None): Handles the mouse press event.
        mouseMoveEvent(event): Handles the mouse move event.
        mouseReleaseEvent(event): Handles the mouse release event.
        updateBoundingBox(pos): Updates the bounding box during mouse drag.
    """

    # Class for creating QGraphicsScene which loads, displays and manipulates
    # images for defining area of interest

    coordinatesChanged = pyqtSignal(QPoint)
    rectDrawn = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self._zoom = 0
        self._pinned = False
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._photo.setShapeMode(
            QGraphicsPixmapItem.BoundingRectShape)
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)
        self.setCursor(Qt.CrossCursor)  # Set default cursor to crosshair
        self.clickedCounter = 0
        self.areaOfInterestCoords = []

        self.flag01 = False  # Has the rect been drawn by clicking
        self.flag02 = False  # Has the rect been drawn by manual input
        self.flag03 = False  # Need to remove drawn rect in place of man rect
        self.flag04 = False  # Need to remove man rect in place of drawn rect

    def hasPhoto(self):
        return not self._empty

    def resetView(self, scale=1):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if (scale := max(1, scale)) == 1:
                self._zoom = 0
            if self.hasPhoto():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height()) * scale
                self.scale(factor, factor)
                if not self.zoomPinned():
                    self.centerOn(self._photo)
                self.updateCoordinates()

    def setPhoto(self, pixmap=None):
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QPixmap())
        if not (self.zoomPinned() and self.hasPhoto()):
            self._zoom = 0
        self.resetView(SCALE_FACTOR ** self._zoom)

    def zoomLevel(self):
        return self._zoom

    def zoomPinned(self):
        return self._pinned

    def setZoomPinned(self, enable):
        self._pinned = bool(enable)

    def zoom(self, step):
        zoom = max(0, self._zoom + (step := int(step)))
        if zoom != self._zoom:
            self._zoom = zoom
            if self._zoom > 0:
                if step > 0:
                    factor = SCALE_FACTOR ** step
                else:
                    factor = 1 / SCALE_FACTOR ** abs(step)
                self.scale(factor, factor)
            else:
                self.resetView()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.zoom(delta and delta // abs(delta))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resetView()

    def toggleDragMode(self):
        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.CrossCursor)

    def updateCoordinates(self, pos=None):
        if self._photo.isUnderMouse():
            if pos is None:
                pos = self.mapFromGlobal(QCursor.pos())
            point = self.mapToScene(pos).toPoint()
        else:
            point = QPoint()
        self.coordinatesChanged.emit(point)

    def leaveEvent(self, event):
        self.coordinatesChanged.emit(QPoint())
        super().leaveEvent(event)

    def enterEvent(self, event):
        super().enterEvent(event)
        self.viewport().setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event, pos=None):
        modifiers = QApplication.keyboardModifiers()
        if event.button() == Qt.LeftButton and modifiers != Qt.ShiftModifier:
            self.viewport().setCursor(Qt.CrossCursor)
            if self._photo.isUnderMouse():
                if pos is None:
                    pos = self.mapFromGlobal(QCursor.pos())
                point = self.mapToScene(pos).toPoint()

                self.clickedCounter += 1
                if self.clickedCounter == 1:
                    self.boxPoint1x = point.x()
                    self.boxPoint1y = point.y()
                    if self.flag03 and self.flag04:
                        self._scene.removeItem(self.rect_item)
                        self.clickedCounter = 0
                        self.flag01 = False
                        self.flag03 = False
                        self.flag02 = True
                elif self.clickedCounter == 2:
                    self.flag01 = True
                    self.boxPoint2x = point.x()
                    self.boxPoint2y = point.y()

                    xaxis = min(self.boxPoint2x, self.boxPoint1x)
                    yaxis = min(self.boxPoint2y, self.boxPoint1y)
                    width = abs(self.boxPoint2x - self.boxPoint1x)
                    height = abs(self.boxPoint2y - self.boxPoint1y)
                    self.areaOfInterestCoords = [xaxis, yaxis, width, height]
                    pen = QPen()
                    pen.setColor(Qt.red)
                    pen.setWidth(3)
                    brush = QBrush()
                    brush.setColor(QColor(255, 0, 0, 80))
                    brush.setStyle(Qt.SolidPattern)
                    self.rect_item = QGraphicsRectItem(
                        QRectF(xaxis, yaxis, width, height))
                    self.rect_item.setPen(pen)
                    self.rect_item.setBrush(brush)
                    self._scene.addItem(self.rect_item)
                    self.rectDrawn.emit()
                    self._scene.removeItem(self.bounding_rect_item)
                elif self.clickedCounter > 2:
                    self.clickedCounter = 0
                    self._scene.removeItem(self.rect_item)
        elif event.button() == Qt.LeftButton and modifiers == Qt.ShiftModifier:
            self.viewport().setCursor(Qt.OpenHandCursor)
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            self.viewport().setCursor(Qt.OpenHandCursor)
        else:
            self.viewport().setCursor(Qt.CrossCursor)
            if self.clickedCounter == 1:
                self.updateBoundingBox(event.pos())
        self.updateCoordinates(event.pos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            self.viewport().setCursor(Qt.OpenHandCursor)
        else:
            self.viewport().setCursor(Qt.CrossCursor)
        super().mouseReleaseEvent(event)

    def updateBoundingBox(self, pos):
        point = self.mapToScene(pos).toPoint()
        xaxis = min(point.x(), self.boxPoint1x)
        yaxis = min(point.y(), self.boxPoint1y)
        width = abs(point.x() - self.boxPoint1x)
        height = abs(point.y() - self.boxPoint1y)
        if hasattr(self, 'bounding_rect_item') and self.bounding_rect_item.scene() == self._scene:
            self._scene.removeItem(self.bounding_rect_item)
        pen = QPen()
        pen.setColor(Qt.red)
        pen.setWidth(2)
        self.bounding_rect_item = QGraphicsRectItem(
            QRectF(xaxis, yaxis, width, height))
        self.bounding_rect_item.setPen(pen)
        self._scene.addItem(self.bounding_rect_item)


class mainProgram(QMainWindow, Ui_MainWindow):
    """mainProgram is a class that inherits from QMainWindow and Ui_MainWindow. 
    It serves as the main class for the GUI application, handling initialization, UI setup, 
    and various functionalities such as settings configuration, image set management, ROI 
    definition, analysis, and results display.
    Methods:
        __init__(self, parent=None): Initializes the mainProgram class.
        settings(self): Configures and displays the settings UI for the application.
        changedSettings(self): Updates the settings based on user input from the settings UI.
        imageSet(self): Sets up the UI for the image set configuration and initializes the relevant settings.
        changedImageSet(self): Updates the image set based on user input and refreshes the displayed image list.
        setMaxTargetImage(self): Sets the maximum target image based on the number of files in the image folder.
        roiDef(self): Sets up the Region of Interest (ROI) definition interface.
        toggleROI(self): Toggles the enabled state of the ROI definition UI elements.
        saveRect(self): Saves the coordinates of the region of interest (ROI) from the roiViewer to the ROI definition UI fields.
        enterManualROI(self): Manually enters the Region of Interest (ROI) coordinates and updates the ROI viewer.
        analysis(self): Sets up the analysis UI within the main frame layout.
        changedAnalysis(self): Updates the analysis settings based on user input from the UI.
        submitDIC(self): Handles the submission of Digital Image Correlation (DIC) analysis.
        stopDIC(self): Stops the DIC analysis and updates the UI accordingly.
        results(self): Sets up the results UI within the main frame layout.
        getImagePairList(self): Reads and unpacks image pair data from a file using MessagePack.
        resultsSelChanged(self): Handles the event when the selection in the results UI changes.
        drawResultsSum(self): Sets up and displays the results summary UI.
        resultsSumChanged(self): Updates the state of various result summary options based on the current UI inputs.
        exportData(self): Exports the displacement and/or strain data to a CSV file.
        drawResultsCon(self): Sets up and populates the results container tab in the GUI.
        resultsConChanged(self): Updates the configuration parameters for the results based on the current UI input values.
        drawResultsCut(self): Configures and populates the UI elements for the results cut tab in the application.
        resultsCutChanged(self): Updates the attributes related to the results cut based on the current UI input values.
        submitGraph(self): Handles the submission of the graph based on selected options.
        plotContourDisp(self): Plots a contour displacement map based on the selected parameters from the UI.
        plotCutLineDisp(self): Plots the displacement cut line based on the user-selected parameters.
        plotContourStrain(self): Plots the contour strain based on the user-selected parameters.
        plotCutLineStrain(self): Plots the strain cut line based on the user-selected parameters.
        anew(self): Creates a new settings file by resetting all settings to their default values.
        asave(self): Saves the current settings to a file.
        asaveAs(self): Saves the current settings to a file with a specified name.
        aopen(self): Opens a settings file and loads the settings.
        aexit(self): Exits the application.
        agithub(self): Opens the GitHub page for the application.
        aversion(self): Displays the current version of the application.
        setDefaultsSettings(self): Resets the settings to their default values.
        setDefaultsImageSet(self): Resets the image set settings to their default values.
        deleteLayout(self, layout): Deletes the specified layout.
        handleCoords(self, point): Handles the display of coordinates in the ROI viewer.
        handleOpen(self): Handles the opening of files.
        openImageSetFolder(self): Opens a dialog to select the image set folder.
        run_planarDICLocal(self, settings, results_file): Runs the planar DIC analysis locally.
        started_progOut(self, text): Handles the start of the program output.
        finished_progOut(self, text): Handles the end of the program output.
        update_progOut(self, text): Updates the program output.
        showUnsaved(self): Displays an indicator for unsaved changes.
        closeEvent(self, event): Handles the close event for the application.
    """

    # Main class for the program

    def __init__(self, parent=None):
        """
        Initializes the mainProgram class.
        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        Attributes:
            _imageFolder (str): Path to the image folder.
            _savePath (str): Path to save files.
            _fileName (str): Name of the file.
            _debugLevel (int): Debug level setting.
            _CPUCount (int): Number of CPU cores to use.
            _DICType (int): Type of DIC (0 for Planar, 1 for Stereo).
            _subSetSize (int): Size of the subset.
            _stepSize (int): Step size.
            _shapeFunc (int): Shape function (0 for Affine, 1 for Quadratic).
            _startingPoints (list): Starting points for the analysis.
            _refStrat (int): Reference strategy (0 for Relative, 1 for Absolute).
            _gausBlurSize (int): Gaussian blur size.
            _gausBlurSTD (float): Gaussian blur standard deviation.
            _datumImage (int): Datum image index.
            _targetImage (int): Target image index.
            _increment (int): Increment value.
            _ROI (list): Region of interest.
            _backgroundCutOff (float): Background cutoff value.
            _optAlgor (int): Optimization algorithm (0 for IC-GN, 1 for IC-LM, 2 for Fast-IC-LM).
            _maxIter (int): Maximum number of iterations.
            _convTol (float): Convergence tolerance.
            _znccTol (float): ZNCC tolerance for convergence.
            flag00 (bool): Indicates if settings have been changed since the last save.
            flag01 (bool): Indicates if the file has results.
            flag02 (bool): Tracks the state of the ROI painter.
            flag03 (bool): Allows saving of settings even if results exist when DIC job is being submitted.
            flag04 (bool): Tracks if the results tab has been opened previously.
            flag05 (bool): Tracks if the user has been on the results - summary tab.
            flag06 (bool): Tracks if the user has been on the results - contour graph tab.
            flag07 (bool): Tracks if the user has been on the results - cutline graph tab.
            flag08 (bool): Used to track if the user has changed the default value for the image set final image.
            flag09 (bool): Spare flag.
            flag10 (bool): Spare flag.
        """

        # Inherit from the aforementioned class and set up the gui
        super(mainProgram, self).__init__(parent)
        self.setupUi(self)

        self.show()

        # Connect all buttons to their respective functions
        self.settingsBut.clicked.connect(self.settings)
        self.imageSetBut.clicked.connect(self.imageSet)
        self.roiBut.clicked.connect(self.roiDef)
        self.analysisBut.clicked.connect(self.analysis)
        self.resultsBut.clicked.connect(self.results)
        self.actionSave_as.triggered.connect(self.asaveAs)
        self.actionSave.triggered.connect(self.asave)
        self.actionOpen.triggered.connect(self.aopen)
        self.actionExit.triggered.connect(self.aexit)
        self.actionNew.triggered.connect(self.anew)
        self.actionGitHub.triggered.connect(self.agithub)
        self.actionVersion.triggered.connect(self.aversion)

        # Disable the results button until results are available
        self.resultsBut.setEnabled(False)

        # ============== Paths ============== #
        self._imageFolder = QStandardPaths.standardLocations(
            QStandardPaths.PicturesLocation)[0]
        self._savePath = None
        self._fileName = None

        # ============== DIC Settings ============== #
        self._debugLevel = None
        self._imageFolder = None
        self._CPUCount = None
        self._DICType = None
        self._subSetSize = None
        self._stepSize = None
        self._shapeFunc = None
        self._startingPoints = None
        self._refStrat = None
        self._gausBlurSize = None
        self._gausBlurSTD = None
        self._datumImage = None
        self._targetImage = None
        self._increment = None
        self._ROI = None
        self._backgroundCutOff = None
        self._optAlgor = None
        self._maxIter = None
        self._convTol = None
        self._znccTol = None  # ZNCC tolerance for convergence

        defSettings = sdset.Settings()
        self._defaultSettings = defSettings

        # ============== Flags ============= #

        # Flags for tracking the status of the program
        self.flag00 = False  # Have the settings been changed since the last save
        self.flag01 = False  # Does the file have results
        self.flag02 = False  # Used for tracking the state of the ROI painter
        # Used when the DIC job is being submitted. Allows the saving of settings even if results exist because they are going to be overwritten with new results.
        self.flag03 = False
        self.flag04 = False  # Used to track if the results tab has been opened previously
        self.flag05 = False  # Used to track if user has been on the results - summary tab
        self.flag06 = False  # Used to track if user has been on the results - contour graph tab
        self.flag07 = False  # Used to track if the user has been on the results - cutline graph tab
        # Used to track if the user has changed the default value for the image set final image
        self.flag08 = False

        # Spare Flags
        self.flag09 = False  #
        self.flag10 = False  #

        # Setting default values - NB: Some things need 'translation' as the GUI uses indexes for some items and the settings class uses strings
        self._debugLevel = self._defaultSettings.DebugLevel
        self._imageFolder = self._defaultSettings.ImageFolder
        self._CPUCount = self._defaultSettings.CPUCount
        #
        if self._defaultSettings.DICType == 'Planar':
            self._DICType = 0
        elif self._defaultSettings.DICType == 'Stereo':
            self._DICType = 1
        self._subSetSize = self._defaultSettings.SubsetSize
        self._stepSize = self._defaultSettings.StepSize
        #
        if self._defaultSettings.ShapeFunctions == 'Affine':
            self._shapeFunc = 0
        elif self._defaultSettings.ShapeFunctions == 'Quadratic':
            self._shapeFunc = 1
        self._startingPoints = self._defaultSettings.StartingPoints
        #
        if self._defaultSettings.ReferenceStrategy == 'Relative':
            self._refStrat = 0
        elif self._defaultSettings.ReferenceStrategy == 'Absolute':
            self._refStrat = 1
        self._gausBlurSize = self._defaultSettings.GaussianBlurSize
        self._gausBlurSTD = self._defaultSettings.GaussianBlurStdDev
        self._datumImage = self._defaultSettings.DatumImage + 1
        self._targetImage = self._defaultSettings.TargetImage
        self._increment = self._defaultSettings.Increment
        self._ROI = self._defaultSettings.ROI
        self._backgroundCutOff = self._defaultSettings.BackgroundCutoff
        #
        if self._defaultSettings.OptimizationAlgorithm == 'IC-GN':
            self._optAlgor = 0
        elif self._defaultSettings.OptimizationAlgorithm == 'IC-LM':
            self._optAlgor = 1
        elif self._defaultSettings.OptimizationAlgorithm == 'Fast-IC-LM':
            self._optAlgor = 2

        self._maxIter = self._defaultSettings.MaxIterations
        self._convTol = self._defaultSettings.ConvergenceThreshold
        # ZNCC tolerance for convergence
        self._znccTol = self._defaultSettings.NZCCThreshold

    def settings(self):
        """
        Configures and displays the settings UI for the application.
        This method sets up the settings tab in the main frame, initializes the UI elements,
        sets their default values, adds validators, connects signals to slots, and adds tooltips.
        The settings tab includes the following elements:
        - Subset size input field
        - Step size input field
        - Starting points input field
        - Maximum iterations input field
        - DIC type dropdown
        - Shape function dropdown
        - Convergence threshold input field
        - Reference strategy dropdown
        - Optimization algorithm dropdown
        - Set Defaults button
        Validators are added to ensure the input fields receive valid data:
        - Subset size and step size must be odd numbers.
        - Starting points and maximum iterations must be positive integers.
        - Convergence threshold must be a positive double.
        Tooltips are provided for each input field to guide the user.
        Signals are connected to the `changedSettings` method to save user input when the input fields are edited or dropdown selections are changed.
        If the main frame layout already exists, it is deleted before creating a new one.
        The Set Defaults button is connected to the `setDefaultsSettings` method to reset the settings to their default values.
        """

        # If the main frame layout exists, delete it. Prevents multiple layouts from being created.
        if hasattr(self, 'mainFrameLayout'):
            if self.mainFrame.layout() is not None:
                self.deleteLayout(self.mainFrameLayout)

        # START OF ACTUAL METHOD
        # Creating a new vertical box layout for the main frame
        self.mainFrameLayout = QVBoxLayout(self.mainFrame)
        self.settingsUI = settingsUI(self)
        self.mainFrameLayout.addWidget(self.settingsUI)

        # Setting values for the settings tab
        self.settingsUI.subsetSizeIn.setText(str(self._subSetSize))
        self.settingsUI.stepSizeIn.setText(str(self._stepSize))
        self.settingsUI.startingPIn.setText(str(self._startingPoints))
        self.settingsUI.maxItIn.setText(str(self._maxIter))
        self.settingsUI.dicTypeBox.setCurrentIndex(self._DICType)
        self.settingsUI.shapeFuncBox.setCurrentIndex(self._shapeFunc)
        self.settingsUI.convergenceIn.setText(str(self._convTol))
        # ZNCC tolerance for convergence
        self.settingsUI.znccTolIn.setText(str(self._znccTol))
        self.settingsUI.refBox.setCurrentIndex(self._refStrat)
        self.settingsUI.algoTypeBox.setCurrentIndex(self._optAlgor)

        # Adding Set Defaults button and connecting it to the setDefaultsSettings method
        self.defaultsBut = QPushButton(self)
        self.defaultsBut.setText('Set Defaults')
        self.mainFrameLayout.addWidget(self.defaultsBut)
        self.defaultsBut.clicked.connect(self.setDefaultsSettings)
        spacer_item = QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.mainFrameLayout.addItem(spacer_item)

        # Adding Validators
        self.settingsUI.subsetSizeIn.setValidator(OddNumberValidator())
        self.settingsUI.stepSizeIn.setValidator(PositiveIntValidator())
        self.settingsUI.startingPIn.setValidator(PositiveIntValidator())
        self.settingsUI.maxItIn.setValidator(PositiveIntValidator())
        self.settingsUI.convergenceIn.setValidator(PositiveDoubleValidator())

        # Connecting the input fields to the changedSettings method to save the user input
        self.settingsUI.subsetSizeIn.editingFinished.connect(
            self.changedSettings)
        self.settingsUI.stepSizeIn.editingFinished.connect(
            self.changedSettings)
        self.settingsUI.startingPIn.editingFinished.connect(
            self.changedSettings)
        self.settingsUI.maxItIn.editingFinished.connect(self.changedSettings)
        self.settingsUI.dicTypeBox.currentIndexChanged.connect(
            self.changedSettings)
        self.settingsUI.shapeFuncBox.currentIndexChanged.connect(
            self.changedSettings)
        self.settingsUI.convergenceIn.editingFinished.connect(
            self.changedSettings)
        self.settingsUI.refBox.currentIndexChanged.connect(
            self.changedSettings)
        self.settingsUI.algoTypeBox.currentIndexChanged.connect(
            self.changedSettings)

        # Adding tooltips
        self.settingsUI.subsetSizeIn.setToolTip(
            "The size of the subset used for the DIC analysis, must be larger than or equal to 1 and odd")
        self.settingsUI.stepSizeIn.setToolTip(
            "The step size used for the DIC analysis, must be larger than or equal to 1")
        self.settingsUI.startingPIn.setToolTip(
            "The number of starting points used for the DIC analysis. The total number of points will be the number of starting points squared. Must be larger than or equal to 1")
        self.settingsUI.maxItIn.setToolTip(
            """The maximum number of iterations used for the optimization algorithm. Must be larger than or equal to 1. 
The default value is set conservatively high and should only be changed when instructed by the code.""")
        self.settingsUI.dicTypeBox.setToolTip(
            "The type of DIC analysis to be performed. Currently only planar is available")
        self.settingsUI.shapeFuncBox.setToolTip(
            "Subset shape functions to use")
        self.settingsUI.convergenceIn.setToolTip(
            "The convergence threshold for the optimization algorithm. Must be larger than 0")
        self.settingsUI.refBox.setToolTip("""The reference strategy to use. Absolute - The reference image is the first image - no change in ROI. Useful for small deformations.
Relative - The reference image is the previous image - ROI is updated for each image pair.  Useful for large deformations.""")
        self.settingsUI.algoTypeBox.setToolTip("""The optimization algorithm to use. IC-GN - Use the Incremental Gauss Newton algorithm.
IC-LM - Use the Incremental Levenberg-Marquardt algorithm. Fast-IC-LM - Use the Fast Incremental Levenberg-Marquardt algorithm.""")

    def changedSettings(self):  # Saving User Input
        """
        Updates the settings based on user input from the settings UI.

        This method retrieves the current values from various input fields in the settings UI
        and updates the corresponding attributes of the class instance. It also sets a flag
        to indicate that there are unsaved changes and triggers the display of an unsaved changes
        indicator.

        Attributes updated:
            _DICType (int): Index of the selected DIC type from dicTypeBox.
            _shapeFunc (int): Index of the selected shape function from shapeFuncBox.
            _subSetSize (str): Text input for the subset size.
            _stepSize (str): Text input for the step size.
            _refStrat (int): Index of the selected reference strategy from refBox.
            _optAlgor (int): Index of the selected optimization algorithm from algoTypeBox.
            _maxIter (str): Text input for the maximum number of iterations.
            _convTol (str): Text input for the convergence tolerance.
            _znccTol (str): Text input for the ZNCC tolerance for convergence.
            _startingPoints (str): Text input for the starting points.

        Sets:
            flag00 (bool): Flag indicating that there are unsaved changes.

        Calls:
            showUnsaved(): Method to display an indicator for unsaved changes.
        """
        self._DICType = self.settingsUI.dicTypeBox.currentIndex()
        self._shapeFunc = self.settingsUI.shapeFuncBox.currentIndex()
        self._subSetSize = (self.settingsUI.subsetSizeIn.text())
        self._stepSize = (self.settingsUI.stepSizeIn.text())
        self._refStrat = self.settingsUI.refBox.currentIndex()
        self._optAlgor = self.settingsUI.algoTypeBox.currentIndex()
        self._maxIter = (self.settingsUI.maxItIn.text())
        self._convTol = (self.settingsUI.convergenceIn.text())
        # ZNCC tolerance for convergence
        self._znccTol = (self.settingsUI.znccTolIn.text())
        self._startingPoints = (self.settingsUI.startingPIn.text())
        self.flag00 = True
        self.showUnsaved()

    def imageSet(self):
        """
        Sets up the UI for the image set configuration and initializes the relevant settings.
        This method performs the following steps:
        1. Deletes the existing layout if it exists.
        2. Creates a new vertical box layout for the main frame.
        3. Initializes the image set UI and adds it to the main frame layout.
        4. Sets default folder display and image folder if they match the default settings.
        5. Updates the UI fields with the current settings for datum image, target image, increment, Gaussian blur size, and background cut-off.
        6. Lists the files in the image folder and sorts them naturally.
        7. Populates the image list in the UI with the sorted files based on the start, end, and increment settings.
        8. Adds validators to the input fields to ensure valid user input.
        9. Connects buttons and input fields to their respective methods for handling user actions and input changes.
        10. Adds tooltips to the input fields to provide additional information to the user.
        Validators:
        - startIn, endIn, incIn: Positive integer validator.
        - gausIn: Odd number validator.
        - backIn: Positive double validator.
        Tooltips:
        - startIn: The image number of the first image in the set. Starts from 1.
        - endIn: The image number of the last image in the set. If left to the default of -1, will use all images after the start image.
        - incIn: The increment between images to use in the analysis.
        - gausIn: The size of the Gaussian blur to apply to the images. Must be larger than or equal to 1 and odd.
        - backIn: Cutoff value to detect all black background in image. This value will be used to detect all black (< Cutoff) areas in the image. 
                  This is useful for automatically removing unwanted areas from the image, e.g., hole in the sample. However, the background MUST be black.
                  Must be larger than 0.
        """

        # If the main frame layout exists, delete it. Prevents multiple layouts from being created.
        if hasattr(self, 'mainFrameLayout'):
            if self.mainFrame.layout() is not None:
                self.deleteLayout(self.mainFrameLayout)

        # START OF ACTUAL METHOD
        # Creating a new vertical box layout for the main frame
        self.mainFrameLayout = QVBoxLayout(self.mainFrame)
        self.imageSetUi = imageSetUi(self)
        self.mainFrameLayout.addWidget(self.imageSetUi)

        # Checks if the default image folder is used and sets the image folder to the default Pictures folder if it is
        if self._imageFolder == self._defaultSettings.ImageFolder:
            self.imageSetUi.folderDisp.setText(QStandardPaths.standardLocations(
                QStandardPaths.PicturesLocation)[0])
            self._imageFolder = QStandardPaths.standardLocations(
                QStandardPaths.PicturesLocation)[0]

        # Grabbing the files in the image folder
        files = os.listdir(self._imageFolder)
        # Do a natural sort on the filenames and display them in the image list
        files = ns.os_sorted(files)
        self.imageSetUi.model = QStandardItemModel()
        self.imageSetUi.dispImages.setModel(self.imageSetUi.model)
        start = int(self._datumImage) - 1
        end = int(self._targetImage)
        if end == -1:
            end = None
        inc = int(self._increment)
        files = files[start:end:inc]
        for i in files:
            item = QStandardItem(i)
            self.imageSetUi.model.appendRow(item)

        if int(self._targetImage) == -1:
            self._targetImage = len(files)

        # Setting values for the image set tab
        self.imageSetUi.startIn.setText(str(self._datumImage))
        self.imageSetUi.endIn.setText(str(self._targetImage))
        self.imageSetUi.incIn.setText(str(self._increment))
        self.imageSetUi.gausIn.setText(str(self._gausBlurSize))
        self.imageSetUi.backIn.setText(str(self._backgroundCutOff))
        self.imageSetUi.folderDisp.setText(self._imageFolder)

        # Adding validators
        self.imageSetUi.startIn.setValidator(PositiveIntValidator())
        self.imageSetUi.endIn.setValidator(PositiveIntValidator())
        self.imageSetUi.incIn.setValidator(PositiveIntValidator())
        self.imageSetUi.gausIn.setValidator(OddNumberValidator())
        self.imageSetUi.backIn.setValidator(Int255Validator())

        # Connecting everything
        self.imageSetUi.defaultsBut.clicked.connect(self.setDefaultsImageSet)
        self.imageSetUi.selFolderBut.clicked.connect(self.openImageSetFolder)
        self.imageSetUi.setMax.clicked.connect(self.setMaxTargetImage)

        # Connecting the input fields to the changedImageSet method to save the user input
        self.imageSetUi.startIn.editingFinished.connect(self.changedImageSet)
        self.imageSetUi.endIn.editingFinished.connect(self.changedImageSet)
        self.imageSetUi.incIn.editingFinished.connect(self.changedImageSet)
        self.imageSetUi.gausIn.editingFinished.connect(self.changedImageSet)
        self.imageSetUi.backIn.editingFinished.connect(self.changedImageSet)

        # Adding tooltips
        self.imageSetUi.startIn.setToolTip(
            "The image number of the first image in the set. Starts from 1.")
        self.imageSetUi.endIn.setToolTip("""The image number of the last image in the set. 
By default it will use all images in the set.""")
        self.imageSetUi.incIn.setToolTip(
            "The increment between images to use in the analysis")
        self.imageSetUi.gausIn.setToolTip(
            "The size of the Gaussian blur to apply to the images. Must be larger than or equal to 1 and odd.")
        self.imageSetUi.backIn.setToolTip("""Cutoff value to detect all black background in image.
This value will be used to detect all black (< Cutoff) areas in the image. 
This is useful for automatically removing unwanted areas from the image, 
eg hole in the sample. However, the background MUST be black.
Must be larger than 0.""")

    def changedImageSet(self):  # Saving User Input
        """
        Updates the image set based on user input and refreshes the displayed image list.
        This method retrieves user input for the starting image, ending image, increment value,
        Gaussian blur size, and background cut-off value. It then updates the internal variables
        with these values. The method also re-fetches the list of image files from the specified
        image folder, sorts them naturally, and updates the displayed image list in the UI based
        on the user-specified range and increment.
        Attributes:
            _datumImage (str): The starting image index as input by the user.
            _targetImage (str): The ending image index as input by the user.
            _increment (str): The increment value for selecting images as input by the user.
            _gausBlurSize (str): The Gaussian blur size as input by the user.
            _backgroundCutOff (str): The background cut-off value as input by the user.
            _imageFolder (str): The folder containing the image files.
            imageSetUi (object): The UI object containing the image set input fields and display.
        Actions:
            Updates the internal variables with user input.
            Re-fetches and sorts the list of image files.
            Updates the displayed image list in the UI.
            Sets the flag00 attribute to True.
            Calls the showUnsaved method to indicate unsaved changes.
        """
        self._datumImage = self.imageSetUi.startIn.text()
        self._targetImage = self.imageSetUi.endIn.text()
        self._increment = self.imageSetUi.incIn.text()
        self._gausBlurSize = self.imageSetUi.gausIn.text()
        self._backgroundCutOff = self.imageSetUi.backIn.text()

        # Doing the grab and sort again to update the image list
        files = os.listdir(self._imageFolder)
        # Do a natural sort on the filenames
        files = ns.os_sorted(files)
        self.imageSetUi.model = QStandardItemModel()
        self.imageSetUi.dispImages.setModel(self.imageSetUi.model)
        start = int(self._datumImage) - 1
        end = int(self._targetImage)
        if end == -1:
            end = None
        inc = int(self._increment)
        files = files[start:end:inc]
        for i in files:
            item = QStandardItem(i)
            self.imageSetUi.model.appendRow(item)
        self.flag00 = True
        self.showUnsaved()

    def setMaxTargetImage(self):
        """
        Sets the maximum target image based on the number of files in the image folder.

        This method lists all files in the specified image folder, counts them, and sets the 
        target image to the total number of files. It then updates the UI element to reflect 
        this number and triggers any necessary changes in the image set.

        Attributes:
            self._imageFolder (str): The path to the folder containing the images.
            self._targetImage (int): The total number of images in the folder.
            self.imageSetUi.endIn (QLineEdit): The UI element displaying the end image number.
        """
        files = os.listdir(self._imageFolder)
        self._targetImage = len(files)
        self.imageSetUi.endIn.setText(str(self._targetImage))
        self.changedImageSet()

    def roiDef(self):
        """
        Sets up the Region of Interest (ROI) definition interface.
        This method initializes the layout and widgets for defining the ROI on an image.
        It handles the following tasks:
        - Deletes the existing layout if it exists.
        - Sets the image folder to the default pictures location if it hasn't been set.
        - Creates and adds the ROI definition UI and photo viewer to the main frame layout.
        - Connects signals for handling coordinate changes and rectangle drawing.
        - Sets up the label for displaying coordinates.
        - Connects the checkbox state change signal to toggle the ROI.
        - Loads the images from the image folder and sets the ROI image to the selected datum image.
        - Sets the initial values for the ROI input fields.
        - Connects the editing finished signals of the ROI input fields to handle manual ROI entry.
        - Draws the ROI if it has already been defined.
        - Adds validators to the ROI input fields to ensure positive integer values.
        - Adds tooltips to the photo viewer for user guidance.
        Attributes:
            mainFrameLayout (QVBoxLayout): The layout for the main frame.
            roiDefUi (roiDefUi): The UI for defining the ROI.
            roiViewer (PhotoViewer): The photo viewer for displaying the image.
            labelCoords (QLabel): The label for displaying coordinates.
            roiImage (str): The filename of the ROI image.
        """
        # If the main frame layout exists, delete it. Prevents multiple layouts from being created.
        if hasattr(self, 'mainFrameLayout'):
            if self.mainFrame.layout() is not None:
                self.deleteLayout(self.mainFrameLayout)

        # This is added here in case the user clicks on the ROI Def tab before going to the Image Set tab
        if self._imageFolder == self._defaultSettings.ImageFolder:
            self._imageFolder = QStandardPaths.standardLocations(
                QStandardPaths.PicturesLocation)[0]

        # Creating a new vertical box layout for the main frame and creating the UI widget and adding it to the layout
        self.mainFrameLayout = QVBoxLayout(self.mainFrame)
        self.roiDefUi = roiDefUi(self)
        self.mainFrameLayout.addWidget(self.roiDefUi)
        self.roiViewer = PhotoViewer(self)
        self.mainFrameLayout.addWidget(self.roiViewer)

        # Connecting signals
        self.roiViewer.coordinatesChanged.connect(self.handleCoords)
        self.roiViewer.rectDrawn.connect(self.saveRect)
        self.labelCoords = QLabel(self)
        self.labelCoords.setAlignment(
            Qt.AlignRight | Qt.AlignCenter)
        self.mainFrameLayout.addWidget(self.labelCoords)
        self.roiDefUi.checkBox.stateChanged.connect(self.toggleROI)

        # Grabbing the files in the image folder and setting the ROI image to be the datum image selected by the user
        files = os.listdir(self._imageFolder)
        files = ns.os_sorted(files)
        self.roiImage = files[int(self._datumImage) - 1]
        # Setting the datum photo to be displayed
        self.roiViewer.setPhoto(
            QPixmap(os.path.join(self._imageFolder, self.roiImage)))
        # Setting the initial values for the ROI input fields
        self.roiDefUi.xIn.setText(str(self._ROI[0]))
        self.roiDefUi.yIn.setText(str(self._ROI[1]))
        self.roiDefUi.widthIn.setText(str(self._ROI[2]))
        self.roiDefUi.heightIn.setText(str(self._ROI[3]))
        # Saving user input
        self.roiDefUi.xIn.editingFinished.connect(self.enterManualROI)
        self.roiDefUi.yIn.editingFinished.connect(self.enterManualROI)
        self.roiDefUi.widthIn.editingFinished.connect(self.enterManualROI)
        self.roiDefUi.heightIn.editingFinished.connect(self.enterManualROI)

        # Handle drawing of ROI for if the user has already defined an ROI
        if self.flag02 == True:
            self.enterManualROI()

        # Adding validators
        self.roiDefUi.xIn.setValidator(PositiveIntValidator())
        self.roiDefUi.yIn.setValidator(PositiveIntValidator())
        self.roiDefUi.widthIn.setValidator(PositiveIntValidator())
        self.roiDefUi.heightIn.setValidator(PositiveIntValidator())

        # Adding tooltips
        self.roiViewer.setToolTip("""Shift+Left Click to pan the image.
Use the mouse wheel to zoom in and out.
Left click set the first corner of the ROI.
Left click again to set the opposite corner of the ROI.
Left click a third time to remove the ROI.""")

    def toggleROI(self):
        """
        Toggles the enabled state of the ROI (Region of Interest) definition UI elements
        based on the state of the checkBox.

        If the checkBox is checked, the ROI definition input fields (top-left x and y 
        coordinates, width, and height) are enabled, allowing the user to input values.
        If the checkBox is unchecked, these input fields are disabled, preventing user input.

        This method modifies the following UI elements:
        - topLeftxDisp
        - topLeftyDisp
        - widthDisp
        - heightDisp
        - xIn
        - yIn
        - widthIn
        - heightIn
        """
        if self.roiDefUi.checkBox.isChecked():
            self.roiDefUi.topLeftxDisp.setEnabled(True)
            self.roiDefUi.topLeftyDisp.setEnabled(True)
            self.roiDefUi.widthDisp.setEnabled(True)
            self.roiDefUi.heightDisp.setEnabled(True)
            self.roiDefUi.xIn.setEnabled(True)
            self.roiDefUi.yIn.setEnabled(True)
            self.roiDefUi.widthIn.setEnabled(True)
            self.roiDefUi.heightIn.setEnabled(True)
        else:
            self.roiDefUi.topLeftxDisp.setEnabled(False)
            self.roiDefUi.topLeftyDisp.setEnabled(False)
            self.roiDefUi.widthDisp.setEnabled(False)
            self.roiDefUi.heightDisp.setEnabled(False)
            self.roiDefUi.xIn.setEnabled(False)
            self.roiDefUi.yIn.setEnabled(False)
            self.roiDefUi.widthIn.setEnabled(False)
            self.roiDefUi.heightIn.setEnabled(False)

    def saveRect(self):
        """
        Saves the coordinates of the region of interest (ROI) from the roiViewer to the ROI definition UI fields.

        This method performs the following actions:
        1. Retrieves the coordinates of the area of interest from the roiViewer.
        2. Sets the x, y, width, and height fields in the ROI definition UI with the retrieved coordinates.
        3. Updates the internal _ROI attribute with the current area of interest coordinates.
        4. Sets the flag02 and flag00 attributes to True.
        5. Calls the showUnsaved method to indicate that there are unsaved changes.
        """
        coords = self.roiViewer.areaOfInterestCoords
        self.roiDefUi.xIn.setText(str(coords[0]))
        self.roiDefUi.yIn.setText(str(coords[1]))
        self.roiDefUi.widthIn.setText(str(coords[2]))
        self.roiDefUi.heightIn.setText(str(coords[3]))
        self._ROI = self.roiViewer.areaOfInterestCoords
        self.flag02 = True
        self.flag00 = True
        self.showUnsaved()

    def enterManualROI(self):
        """
        Manually enters the Region of Interest (ROI) coordinates and updates the ROI viewer.

        This method retrieves the ROI coordinates from the user interface, updates the ROI viewer
        with a rectangular item representing the ROI, and sets various flags to indicate the state
        of the ROI selection.

        Attributes:
            roiViewer (object): The viewer object that displays the ROI.
            roiDefUi (object): The user interface object containing input fields for ROI coordinates.
            flag01 (bool): A flag indicating whether the previous ROI rectangle item was removed.
            flag02 (bool): A flag indicating whether the new ROI rectangle item was added.
            flag03 (bool): A flag indicating the state of the ROI viewer.
            flag04 (bool): A flag indicating the state of the ROI viewer.
            flag00 (bool): A flag indicating whether the unsaved changes should be shown.
            _ROI (list): A list containing the coordinates [x, y, width, height] of the ROI.

        Steps:
            1. Check if the ROI viewer already has a rectangle item and remove it if present.
            2. Retrieve the ROI coordinates (x, y, width, height) from the user interface.
            3. Update the internal ROI attribute and the ROI viewer's area of interest coordinates.
            4. Create a new rectangular item with the specified coordinates and add it to the ROI viewer.
            5. Set the appropriate flags to indicate the state of the ROI selection.
            6. If the checkbox in the user interface is checked, set the flag to show unsaved changes.
        """
        if hasattr(self.roiViewer, 'rect_item'):
            self.roiViewer.clickedCounter = 0
            self.roiViewer._scene.removeItem(self.roiViewer.rect_item)
        x = int(self.roiDefUi.xIn.text())
        y = int(self.roiDefUi.yIn.text())
        width = int(self.roiDefUi.widthIn.text())
        height = int(self.roiDefUi.heightIn.text())
        ROI = [x, y, width, height]
        self._ROI = ROI
        self.roiViewer.areaOfInterestCoords = [x, y, width, height]
        self.roiViewer.flag03 = True
        self.roiViewer.flag04 = True
        pen = QPen()
        pen.setColor(Qt.red)
        pen.setWidth(3)
        brush = QBrush()
        brush.setColor(QColor(255, 0, 0, 80))
        brush.setStyle(Qt.SolidPattern)
        self.roiViewer.rect_item = QGraphicsRectItem(
            QRectF(x, y, width, height))
        self.roiViewer.rect_item.setPen(pen)
        self.roiViewer.rect_item.setBrush(brush)
        self.roiViewer._scene.addItem(self.roiViewer.rect_item)
        self.flag02 = True
        if self.roiDefUi.checkBox.isChecked():
            self.flag00 = True
            self.showUnsaved()

    def analysis(self):
        """
        Sets up the analysis UI within the main frame layout. If a layout already exists, it deletes it first.
        Initializes the analysis UI components, sets default values, and connects signals to slots.
        - If `self.flag01` is True, a warning message is displayed indicating that running the analysis will overwrite existing results.
        - Populates the debug level combo box with options and sets the current index to the stored debug level.
        - Sets the CPU count input field to the stored CPU count.
        - Connects the debug level combo box and CPU count input field to their respective change handlers.
        - Adds a validator to the CPU count input field to ensure only positive integers are entered.
        """
        if hasattr(self, 'mainFrameLayout'):
            if self.mainFrame.layout() is not None:
                self.deleteLayout(self.mainFrameLayout)

        self.mainFrameLayout = QVBoxLayout(self.mainFrame)
        self.analysisUi = analysisUi(self)
        self.mainFrameLayout.addWidget(self.analysisUi)
        self.analysisUi.progOut.setText(
            "Program Output will be captured here..........")
        self.analysisUi.startBut.clicked.connect(self.submitDIC)
        self.analysisUi.killBut.clicked.connect(self.stopDIC)
        if self.flag01 == True:
            self.analysisUi.progOut.setText(
                "Results already exist for this file, running the analysis will overwrite the existing results")
        self.analysisUi.debugIn.addItem("")
        self.analysisUi.debugIn.addItem("")
        self.analysisUi.debugIn.addItem("")
        self.analysisUi.debugIn.setItemText(0, "0 - No Debugging")
        self.analysisUi.debugIn.setItemText(1, "1 - Debugging")
        self.analysisUi.debugIn.setItemText(
            2, "2 - Debugging with extra information")
        self.analysisUi.debugIn.setCurrentIndex(self._debugLevel)
        self.analysisUi.cpuIn.setText(str(self._CPUCount))

        # Connecting to save input
        self.analysisUi.debugIn.currentIndexChanged.connect(
            self.changedAnalysis)
        self.analysisUi.cpuIn.editingFinished.connect(self.changedAnalysis)

        # Adding validators
        self.analysisUi.cpuIn.setValidator(PositiveIntValidator())

    def changedAnalysis(self):
        """
        Updates the analysis settings based on user input from the UI.

        This method retrieves the CPU count and debug level from the UI elements,
        updates the corresponding instance variables, sets a flag to indicate that
        changes have been made, and triggers the display of an unsaved changes indicator.

        Attributes:
            self._CPUCount (int): The number of CPUs to be used, retrieved from the UI.
            self._debugLevel (int): The debug level, retrieved from the UI.
            self.flag00 (bool): A flag indicating that changes have been made.
        """
        self._CPUCount = int(self.analysisUi.cpuIn.text())
        self._debugLevel = int(self.analysisUi.debugIn.currentIndex())
        self.flag00 = True
        self.showUnsaved()

    def submitDIC(self):
        """
        Handles the submission of Digital Image Correlation (DIC) analysis.

        This method performs the following steps:
        1. Sets the flag03 attribute to True.
        2. Calls the asave() method to save the current state.
        3. Sets the flag03 attribute to False.
        4. Loads DIC settings from a MsgPack file specified by the _savePath attribute.
        5. Appends the loaded DIC settings to the progOut attribute of the analysisUi object.
        6. Runs the planar DIC analysis using the loaded settings and the _savePath attribute.
        7. Disables several buttons in the UI (settingsBut, imageSetBut, roiBut, analysisBut, resultsBut).
        8. Sets the flag01 attribute to True.
        """
        self.flag03 = True
        self.asave()
        self.flag03 = False
        dicSet = sdset.Settings.fromMsgPackFile(self._savePath)
        self.analysisUi.progOut.append(f"{dicSet}")
        self.run_planarDICLocal(dicSet, self._savePath)
        self.settingsBut.setEnabled(False)
        self.imageSetBut.setEnabled(False)
        self.roiBut.setEnabled(False)
        self.analysisBut.setEnabled(False)
        self.resultsBut.setEnabled(False)
        self.analysisUi.startBut.setEnabled(False)
        self.flag01 = True

    def stopDIC(self):
        self.flag01 = False
        self.analysisUi.progOut.append("Analysis Stopped")
        self.settingsBut.setEnabled(True)
        self.imageSetBut.setEnabled(True)
        self.roiBut.setEnabled(True)
        self.analysisBut.setEnabled(True)
        self.analysisUi.startBut.setEnabled(True)
        subprocess.run(['ray', 'stop'])
        sd.safe_ray_shutdown(externalRay=False)
        self.analysisUi.statusLab.setText("Status: Analysis Stopped")
        self.analysisUi.statusLab.setStyleSheet("color: red")
        self.asave()

    def results(self):
        """
        Sets up the results UI within the main frame layout. If a layout already exists, it deletes it first.
        Initializes the results UI components, sets default values, and connects signals to slots.
        - Retrieves the last image pair from the saved results.
        - Sets the current index of option1 and option2 combo boxes based on the flags.
        - Connects the combo boxes to their respective change handlers.
        """
        # If the main frame layout exists, delete it. Prevents multiple layouts from being created.
        if hasattr(self, 'mainFrameLayout'):
            if self.mainFrame.layout() is not None:
                self.deleteLayout(self.mainFrameLayout)

        # START OF ACTUAL METHOD
        # Creating a new vertical box layout for the main frame
        self.mainFrameLayout = QVBoxLayout(self.mainFrame)
        self.resultsUi = resultsUi(self)
        self.mainFrameLayout.addWidget(self.resultsUi)

        # Getting the last image pair
        self.lastImagePair = self.getImagePairList()

        # Connecting the combo boxes to their respective change handlers
        self.resultsUi.textBut.clicked.connect(self.drawResultsSum)
        self.resultsUi.contBut.clicked.connect(self.drawResultsCon)
        self.resultsUi.lineBut.clicked.connect(self.drawResultsCut)

    def getImagePairList(self):
        """
        Reads and unpacks image pair data from a file using MessagePack.

        This method opens a file specified by the instance's `_savePath` attribute,
        and uses the MessagePack library to unpack data from the file. It skips the
        first three unpacked items, then iteratively unpacks image pairs and their
        corresponding dimensions and data until it encounters an image pair with a
        value of -1 or runs out of data.

        Returns:
            The last unpacked image pair before encountering -1 or running out of data.
        """
        with open(self._savePath, 'rb') as file:
            file.seek(0)
            unpacker = msgpack.Unpacker(file, raw=False)
            _ = unpacker.unpack()
            _ = unpacker.unpack()
            _ = unpacker.unpack()
            try:
                while True:
                    imgPair = unpacker.unpack()
                    dim = unpacker.unpack()
                    data = unpacker.unpack().reshape(dim)
                    if imgPair == -1:
                        break
            except msgpack.OutOfData:
                pass
            return imgPair

    def resultsSelChanged(self):
        """
        Handles the event when the selection in the results UI changes.
        This method updates the indices for option1 and option2 based on the current
        selection in the results UI. It then sets up the font and size policy for the
        UI elements. Depending on the selected index of option1, it calls the appropriate
        method to draw the results.
        - If option1 index is 0, it calls `drawResultsSum()`.
        - If option1 index is 1, it calls `drawResultsCon(font, sizePolicy)`.
        - If option1 index is 2, it calls `drawResultsCut(font, sizePolicy)`.
        Attributes:
            option1Ind (int): The index of the selected option in option1.
            option2Ind (int): The index of the selected option in option2.
        """
        # Saving the current indices
        self.option2Ind = int(self.resultsSelector.currentIndex())
        if self.option1Ind == 0:
            self.drawResultsSum()
        elif self.option1Ind == 1:
            self.drawResultsCon()
        elif self.option1Ind == 2:
            self.drawResultsCut()

    def drawResultsSum(self):
        """
        Sets up and displays the results summary UI.
        This method initializes and configures the results summary user interface,
        including setting default values, connecting signals to slots, and adding
        validators and tooltips to input fields.
        The method performs the following steps:
        1. Deletes the existing temporary layout and creates a new QVBoxLayout.
        2. Adds the new layout to the main vertical layout.
        3. Initializes the results summary UI component and adds it to the temporary layout.
        4. Disables the option2 button in the results UI.
        5. Populates the image pair input dropdown with available image pairs.
        6. Sets default values for the input fields if the flag05 is not set.
        7. Restores previous values for the input fields if the flag05 is set.
        8. Connects input field signals to the resultsSumChanged slot.
        9. Adds validators to the smooth window and smooth order input fields.
        10. Connects the writeDataBut button to the exportData slot.
        11. Sets tooltips for the smooth window and smooth order input fields.
        Note:
            - The smooth window size must be an odd number or zero.
            - The smooth order must be a positive integer.
            - If the smooth window size is greater than zero, it must be larger than the smooth order.
        """
        self.option1Ind = 0
        # Standard layout delete and creation
        self.deleteLayout(self.resultsUi.tempLayout)
        self.resultsUi.tempLayout = QVBoxLayout()
        self.resultsUi.mainVLayout.addLayout(self.resultsUi.tempLayout)
        self.resultsUiSum = resultsUiSum(self)
        self.resultsUi.tempLayout.addWidget(self.resultsUiSum)

        # Populating the image pair input dropdown
        for i in range(0, self.lastImagePair+1):
            self.resultsUiSum.imgPairIn.addItem("")
            self.resultsUiSum.imgPairIn.setItemText(
                i, f"{self.lastImagePair-i}")

        # Setting values for the results summary tab
        if not self.flag05:
            self.resultsUiSum.imgPairIn.setCurrentIndex(0)
            self.resultsUiSum.smoothWindowIn.setText("3")
            self.resultsUiSum.smoothOrderIn.setText("2")
            self.resultsUiSum.removeNanIn.setChecked(True)
            self.resultsUiSum.incDispIn.setChecked(True)
            self.resultsUiSum.incStrainsIn.setChecked(True)
            self.resultsSumChanged()
            self.flag05 = True
        else:
            self.resultsUiSum.imgPairIn.setCurrentIndex(self.resultsSumImgPair)
            self.resultsUiSum.smoothWindowIn.setText(
                self.resultsSumSmoothWindow)
            self.resultsUiSum.smoothOrderIn.setText(self.resultsSumSmoothOrder)
            self.resultsUiSum.removeNanIn.setChecked(self.resultsSumRemoveNan)
            self.resultsUiSum.incDispIn.setChecked(self.resultsSumIncDisp)
            self.resultsUiSum.incStrainsIn.setChecked(self.resultsSumIncStrain)

        # Connecting the input fields to the resultsSumChanged method
        self.resultsUiSum.imgPairIn.currentIndexChanged.connect(
            self.resultsSumChanged)
        self.resultsUiSum.smoothWindowIn.editingFinished.connect(
            self.resultsSumChanged)
        self.resultsUiSum.smoothOrderIn.editingFinished.connect(
            self.resultsSumChanged)
        self.resultsUiSum.removeNanIn.stateChanged.connect(
            self.resultsSumChanged)
        self.resultsUiSum.incDispIn.stateChanged.connect(
            self.resultsSumChanged)
        self.resultsUiSum.incStrainsIn.stateChanged.connect(
            self.resultsSumChanged)

        # Adding validators
        self.resultsUiSum.smoothWindowIn.setValidator(OddNumberZeroValidator())
        self.resultsUiSum.smoothOrderIn.setValidator(PositiveIntValidator())

        # Connecting the writeDataBut button to the exportData method
        self.resultsUiSum.writeDataBut.clicked.connect(self.exportData)

        # Adding tooltips
        self.resultsUiSum.smoothWindowIn.setToolTip("""The size of the window for smoothing the results. Must be an odd number.
If only exporting displacement data, then this value can be 0.
If this value is greater than 0, then it must be larger than the smooth order.""")
        self.resultsUiSum.smoothOrderIn.setToolTip(
            "Order of the Savitzky-Golay smoothing polynomial.")

    def resultsSumChanged(self):
        """
        Updates the state of various result summary options based on the current UI inputs.
        This method retrieves the current values from the UI elements related to result summary
        and updates the corresponding instance variables. It also enables or disables the 
        'writeDataBut' button based on whether include displacement or include strain 
        options are selected.
        Instance Variables Updated:
        - self.resultsSumImgPair: Index of the selected image pair.
        - self.resultsSumSmoothWindow: Text input for the smoothing window.
        - self.resultsSumSmoothOrder: Text input for the smoothing order.
        - self.resultsSumRemoveNan: Boolean indicating whether to remove NaN values.
        - self.resultsSumIncDisp: Boolean indicating whether include displacement is selected.
        - self.resultsSumIncStrain: Boolean indicating whether include strain is selected.
        UI Elements Affected:
        - self.resultsUiSum.writeDataBut: Enabled if either include displacement or include strain is selected, disabled otherwise.
        """
        self.resultsSumImgPair = self.resultsUiSum.imgPairIn.currentIndex()
        self.resultsSumSmoothWindow = self.resultsUiSum.smoothWindowIn.text()
        self.resultsSumSmoothOrder = self.resultsUiSum.smoothOrderIn.text()
        self.resultsSumRemoveNan = self.resultsUiSum.removeNanIn.isChecked()
        self.resultsSumIncDisp = self.resultsUiSum.incDispIn.isChecked()
        self.resultsSumIncStrain = self.resultsUiSum.incStrainsIn.isChecked()

        if self.resultsSumIncDisp or self.resultsSumIncStrain:
            self.resultsUiSum.writeDataBut.setEnabled(True)
        else:
            self.resultsUiSum.writeDataBut.setEnabled(False)

    def exportData(self):
        """
        Exports the displacement and/or strain data to a CSV file.
        This function retrieves the displacement and/or strain data based on the 
        specified parameters, processes the data (including optional smoothing and 
        NaN removal), and then saves the results to a CSV file using a file dialog 
        for user input.
        Exception: If an error occurs during data retrieval, processing, or saving, 
                   an error message is displayed in a popup dialog.
        """
        try:
            # Saving the settings in correct data format
            imgPair = self.lastImagePair - self.resultsSumImgPair
            smoothWindow = int(self.resultsSumSmoothWindow)
            smoothOrder = int(self.resultsSumSmoothOrder)
            removeNan = self.resultsSumRemoveNan
            incDisp = self.resultsSumIncDisp
            incStrain = self.resultsSumIncStrain

            # Getting required data
            if incDisp:
                dispResults, _, _ = sdpp.getDisplacements(
                    self._savePath, imgPair=imgPair, smoothWindow=smoothWindow, smoothOrder=smoothOrder)
            if incStrain:
                strainResults, _, _ = sdpp.getStrains(
                    self._savePath, imgPair=imgPair, smoothWindow=smoothWindow, smoothOrder=smoothOrder)

            # Optional remove NaN
            if removeNan:
                if incDisp:
                    dispResults = dispResults[~np.isnan(
                        dispResults).any(axis=1)]
                if incStrain:
                    strainResults = strainResults[~np.isnan(
                        strainResults).any(axis=1)]

            # Saving the data as data frames
            if incDisp and incStrain:
                dispDataFrame = pd.DataFrame(dispResults, columns=[
                                             "X Coord", "Y Coord", "Z Coord", "X Disp", "Y Disp", "Z Disp", "Disp Magnitude"])
                strainDataFrame = pd.DataFrame(strainResults, columns=[
                                               "X Coord", "Y Coord", "Z Coord", "X Strain Comp", "Y Strain Comp", "XY Strain Comp", "Von Mises Strain"])
                results = pd.merge(dispDataFrame, strainDataFrame, how="right", on=[
                                   "X Coord", "Y Coord", "Z Coord"])
            elif incDisp:
                results = pd.DataFrame(dispResults, columns=[
                                       "X Coord", "Y Coord", "Z Coord", "X Disp", "Y Disp", "Z Disp", "Disp Magnitude"])
            elif incStrain:
                results = pd.DataFrame(strainResults, columns=[
                                       "X Coord", "Y Coord", "Z Coord", "X Strain Comp", "Y Strain Comp", "XY Strain Comp", "Von Mises Strain"])

            # Saving the data to a CSV file
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            savePath, _ = QFileDialog.getSaveFileName(
                self, "Save File", "", "CSV Files (*.csv)", options=options)
            if savePath:
                if not savePath.endswith(".csv"):
                    savePath = savePath + ".csv"
                results.to_csv(savePath, index=False)
        except Exception as e:
            # Capture the standard error and display it in a popup
            error_message = str(e)
            QMessageBox.critical(
                self, "Error", f"An error occurred: {error_message}")

    def drawResultsCon(self):
        """
        Sets up and populates the results container tab in the GUI.
        Args:
            font (QFont): The font to be used for the Submit Graph button.
            sizePolicy (QSizePolicy): The size policy to be applied to the Submit Graph button.
        This method performs the following tasks:
            1. Deletes the existing layout and creates a new temporary layout.
            2. Populates the image pair and component input fields based on user input.
            3. Sets default or previously saved values for the results container tab.
            4. Connects input fields to the resultsConChanged method to handle changes.
            5. Adds a spacer to the layout for better visual organization.
            6. Enables the option2 button.
            7. Creates and adds the Submit Graph button to the layout.
            8. Connects the Submit Graph button to the submitGraph method.
            9. Adds validators to input fields to ensure valid input.
            10. Adds tooltips to input fields to provide additional information to the user.
        """
        self.option1Ind = 1

        # Size Policy and Font
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)

        # Standard delete and create
        self.deleteLayout(self.resultsUi.tempLayout)
        self.resultsUi.tempLayout = QVBoxLayout()
        self.resultsUi.mainVLayout.addLayout(self.resultsUi.tempLayout)
        self.resultsSelector = QComboBox(self)
        self.resultsSelector.addItem("")
        self.resultsSelector.setItemText(0, "Displacement")
        self.resultsSelector.addItem("")
        self.resultsSelector.setItemText(1, "Strain")
        self.resultsUi.tempLayout.addWidget(self.resultsSelector)
        self.resultsSelector.setSizePolicy(sizePolicy)
        self.resultsSelector.setFont(font)

        if not self.flag04:
            self.option2Ind = 0
            self.resultsSelector.setCurrentIndex(self.option2Ind)
            self.flag04 = True
        else:
            self.resultsSelector.setCurrentIndex(self.option2Ind)
        self.resultsSelector.currentIndexChanged.connect(
            self.resultsSelChanged)
        self.resultsUiCon = resultsUiCon(self)
        self.resultsUi.tempLayout.addWidget(self.resultsUiCon)

        # Populating the image pair and component input fields depending on user input
        for i in range(0, self.lastImagePair+1):
            self.resultsUiCon.imgPairIn.addItem("")
            self.resultsUiCon.imgPairIn.setItemText(
                i, f"{self.lastImagePair-i}")
        if int(self.resultsSelector.currentIndex()) == 0:
            for index, e in enumerate(sdpp.DispComp):
                self.resultsUiCon.compIn.addItem("")
                self.resultsUiCon.compIn.setItemText(
                    index, f"{e.display_name}")
        elif int(self.resultsSelector.currentIndex()) == 1:
            for index, e in enumerate(sdpp.StrainComp):
                self.resultsUiCon.compIn.addItem("")
                self.resultsUiCon.compIn.setItemText(
                    index, f"{e.display_name}")

        # Setting values for the results container tab
        if not self.flag06:
            self.resultsUiCon.imgPairIn.setCurrentIndex(0)
            self.resultsUiCon.smoothWinIn.setText("3")
            self.resultsUiCon.alphaIn.setText("0.75")
            self.resultsUiCon.compIn.setCurrentIndex(0)
            self.resultsUiCon.maxValIn.setText("None")
            self.resultsUiCon.minValIn.setText("None")
            self.resultsUiCon.smoothOrderIn.setText("2")
            self.flag06 = True
            self.resultsConChanged()
        else:
            self.resultsUiCon.imgPairIn.setCurrentIndex(self.resultsConImgPair)
            self.resultsUiCon.smoothWinIn.setText(self.resultsConSmoothWin)
            self.resultsUiCon.alphaIn.setText(self.resultsConAlpha)
            self.resultsUiCon.compIn.setCurrentIndex(self.resultsConComp)
            self.resultsUiCon.maxValIn.setText(self.resultsConMaxVal)
            self.resultsUiCon.minValIn.setText(self.resultsConMinVal)
            self.resultsUiCon.smoothOrderIn.setText(self.resultsConSmoothOrder)

        # Connecting the input fields to the resultsConChanged method
        self.resultsUiCon.imgPairIn.currentIndexChanged.connect(
            self.resultsConChanged)
        self.resultsUiCon.smoothWinIn.editingFinished.connect(
            self.resultsConChanged)
        self.resultsUiCon.alphaIn.editingFinished.connect(
            self.resultsConChanged)
        self.resultsUiCon.compIn.currentIndexChanged.connect(
            self.resultsConChanged)
        self.resultsUiCon.maxValIn.editingFinished.connect(
            self.resultsConChanged)
        self.resultsUiCon.minValIn.editingFinished.connect(
            self.resultsConChanged)
        self.resultsUiCon.smoothOrderIn.editingFinished.connect(
            self.resultsConChanged)

        # Creating a spacer to neaten up the layout
        spacer = QSpacerItem(20, 20, QSizePolicy.Minimum,
                             QSizePolicy.MinimumExpanding)
        self.resultsUi.tempLayout.addItem(spacer)

        # Creating and adding the Submit Graph button
        submitGraphBut = QPushButton("Submit Graph", self)
        submitGraphBut.setSizePolicy(sizePolicy)
        submitGraphBut.setFont(font)
        self.resultsUi.tempLayout.insertWidget(
            2, submitGraphBut, 0, QtCore.Qt.AlignHCenter)

        # Connecting the Submit Graph button to the submitGraph method
        submitGraphBut.clicked.connect(self.submitGraph)

        # Adding validators
        self.resultsUiCon.smoothWinIn.setValidator(OddNumberZeroValidator())
        self.resultsUiCon.smoothOrderIn.setValidator(PositiveIntValidator())

        # Adding tooltips
        self.resultsUiCon.alphaIn.setToolTip(
            "The transparency of the contour plot. Must be between 0 and 1.")
        self.resultsUiCon.maxValIn.setToolTip(
            "The maximum value for the contour plot. If left to None, will use the maximum value in the data.")
        self.resultsUiCon.minValIn.setToolTip(
            "The minimum value for the contour plot. If left to None, will use the minimum value in the data.")
        self.resultsUiCon.smoothOrderIn.setToolTip(
            "Order of the Savitzky-Golay smoothing polynomial.")
        self.resultsUiCon.smoothWinIn.setToolTip("""The size of the window for smoothing the results. Must be an odd number.
A value of 0 means no smoothing but can only be set to zero for displacement graphs.""")

    def resultsConChanged(self):
        """
        Updates the configuration parameters for the results based on the current UI input values.

        This method retrieves the current values from the UI components and updates the corresponding
        instance variables. The parameters updated include:
        - Image pair index
        - Smoothing window size
        - Alpha value
        - Comparison index
        - Maximum value
        - Minimum value
        - Smoothing order
        """
        self.resultsConImgPair = self.resultsUiCon.imgPairIn.currentIndex()
        self.resultsConSmoothWin = self.resultsUiCon.smoothWinIn.text()
        self.resultsConAlpha = self.resultsUiCon.alphaIn.text()
        self.resultsConComp = self.resultsUiCon.compIn.currentIndex()
        self.resultsConMaxVal = self.resultsUiCon.maxValIn.text()
        self.resultsConMinVal = self.resultsUiCon.minValIn.text()
        self.resultsConSmoothOrder = self.resultsUiCon.smoothOrderIn.text()

    def drawResultsCut(self):
        """
        Configures and populates the UI elements for the results cut tab in the application.
        Args:
            font (QFont): The font to be used for the UI elements.
            sizePolicy (QSizePolicy): The size policy to be applied to the UI elements.
        This method performs the following tasks:
            1. Deletes the existing layout and creates a new one.
            2. Populates the image pair and component input fields based on user input.
            3. Sets default values for the results cut tab if not already set.
            4. Connects the input fields to the resultsCutChanged method.
            5. Adds a spacer for better visual organization.
            6. Enables the option2 button.
            7. Creates and adds the Submit Graph button, and connects it to the submitGraph method.
            8. Adds validators to the input fields.
            9. Adds tooltips to the input fields for user guidance.
        """
        self.option1Ind = 2

        # Size Policy and Font
        font = QtGui.QFont()
        font.setFamily("Figtree Light")
        font.setPointSize(11)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)

        # Standard layout delete and create
        self.deleteLayout(self.resultsUi.tempLayout)
        self.resultsUi.tempLayout = QVBoxLayout()
        self.resultsUi.mainVLayout.addLayout(self.resultsUi.tempLayout)
        self.resultsSelector = QComboBox(self)
        self.resultsSelector.addItem("")
        self.resultsSelector.setItemText(0, "Displacement")
        self.resultsSelector.addItem("")
        self.resultsSelector.setItemText(1, "Strain")
        self.resultsUi.tempLayout.addWidget(self.resultsSelector)
        self.resultsSelector.setSizePolicy(sizePolicy)
        self.resultsSelector.setFont(font)
        if not self.flag04:
            self.option2Ind = 0
            self.resultsSelector.setCurrentIndex(self.option2Ind)
            self.flag04 = True
        else:
            self.resultsSelector.setCurrentIndex(self.option2Ind)
        self.resultsSelector.currentIndexChanged.connect(
            self.resultsSelChanged)

        self.resultsUiCut = resultsUiCut(self)
        self.resultsUi.tempLayout.addWidget(self.resultsUiCut)

        # Populating the image pair and component input fields depending on user input
        for i in range(0, self.lastImagePair+1):
            self.resultsUiCut.imgPairIn.addItem("")
            self.resultsUiCut.imgPairIn.setItemText(
                i, f"{self.lastImagePair-i}")
        if int(self.resultsSelector.currentIndex()) == 0:
            for index, e in enumerate(sdpp.DispComp):
                self.resultsUiCut.compIn.addItem("")
                self.resultsUiCut.compIn.setItemText(
                    index, f"{e.display_name}")
        elif int(self.resultsSelector.currentIndex()) == 1:
            for index, e in enumerate(sdpp.StrainComp):
                self.resultsUiCut.compIn.addItem("")
                self.resultsUiCut.compIn.setItemText(
                    index, f"{e.display_name}")
        for index, e in enumerate(sdpp.CompID):
            if e.display_name != None:
                self.resultsUiCut.cutCompIn.addItem("")
                self.resultsUiCut.cutCompIn.setItemText(
                    index, f"{e.display_name}")

        # Setting values for the results cut tab
        if not self.flag07:
            self.resultsUiCut.imgPairIn.setCurrentIndex(0)
            self.resultsUiCut.smoothWinIn.setText("3")
            self.resultsUiCut.compIn.setCurrentIndex(0)
            self.resultsUiCut.smoothOrderIn.setText("2")
            self.resultsUiCut.cutCompIn.setCurrentIndex(0)
            self.resultsUiCut.gridLinesIn.setChecked(True)
            self.resultsUiCut.interpIn.setChecked(False)
            self.resultsUiCut.cutValIn.setText("0")
            self.flag07 = True
            self.resultsCutChanged()
        else:
            self.resultsUiCut.imgPairIn.setCurrentIndex(self.resultsCutImgPair)
            self.resultsUiCut.smoothWinIn.setText(self.resultsCutSmoothWin)
            self.resultsUiCut.compIn.setCurrentIndex(self.resultsCutComp)
            self.resultsUiCut.smoothOrderIn.setText(self.resultsCutSmoothOrder)
            self.resultsUiCut.cutCompIn.setCurrentIndex(self.resultsCutCutComp)
            self.resultsUiCut.gridLinesIn.setChecked(self.resultsCutGridLines)
            self.resultsUiCut.interpIn.setChecked(self.resultsCutInterp)
            self.resultsUiCut.cutValIn.setText(self.resultsCutCutValues)

        # Connecting the input fields to the resultsCutChanged method
        self.resultsUiCut.imgPairIn.currentIndexChanged.connect(
            self.resultsCutChanged)
        self.resultsUiCut.smoothWinIn.editingFinished.connect(
            self.resultsCutChanged)
        self.resultsUiCut.compIn.currentIndexChanged.connect(
            self.resultsCutChanged)
        self.resultsUiCut.smoothOrderIn.editingFinished.connect(
            self.resultsCutChanged)
        self.resultsUiCut.cutCompIn.currentIndexChanged.connect(
            self.resultsCutChanged)
        self.resultsUiCut.gridLinesIn.stateChanged.connect(
            self.resultsCutChanged)
        self.resultsUiCut.interpIn.stateChanged.connect(self.resultsCutChanged)
        self.resultsUiCut.cutValIn.editingFinished.connect(
            self.resultsCutChanged)

        # Adding a spacer for better visual organization
        spacer = QSpacerItem(20, 20, QSizePolicy.Minimum,
                             QSizePolicy.MinimumExpanding)
        self.resultsUi.tempLayout.addItem(spacer)

        # Creating and adding the Submit Graph button
        submitGraphBut = QPushButton("Submit Graph", self)
        submitGraphBut.setSizePolicy(sizePolicy)
        submitGraphBut.setFont(font)
        self.resultsUi.tempLayout.insertWidget(
            2, submitGraphBut, 0, QtCore.Qt.AlignHCenter)

        # Connecting the Submit Graph button to the submitGraph method
        submitGraphBut.clicked.connect(self.submitGraph)

        # Adding validators
        self.resultsUiCut.smoothWinIn.setValidator(OddNumberZeroValidator())
        self.resultsUiCut.smoothOrderIn.setValidator(PositiveIntValidator())

        # Adding tooltips
        self.resultsUiCut.smoothWinIn.setToolTip("""The size of the window for smoothing the results. Must be an odd number.
A value of 0 means no smoothing but can only be set to zero for displacement graphs.""")
        self.resultsUiCut.smoothOrderIn.setToolTip(
            "Order of the Savitzky-Golay smoothing polynomial.")
        self.resultsUiCut.cutValIn.setToolTip(
            "The values at which to cut the data. Must be a comma separated list of integers.")
        self.resultsUiCut.gridLinesIn.setToolTip(
            "Whether to show grid lines on the plot.")
        self.resultsUiCut.interpIn.setToolTip(
            "Whether to interpolate the data.")
        self.resultsUiCut.cutCompIn.setToolTip(
            "The component to cut the data along.")
        self.resultsUiCut.compIn.setToolTip("The component to plot.")

    def resultsCutChanged(self):
        """
        Updates the attributes related to the results cut based on the current UI input values.

        This method retrieves the current values from the UI elements and assigns them to the corresponding
        attributes of the class instance. The attributes updated are:
        - resultsCutImgPair: Index of the selected image pair.
        - resultsCutSmoothWin: Text value of the smoothing window input.
        - resultsCutComp: Index of the selected component.
        - resultsCutSmoothOrder: Text value of the smoothing order input.
        - resultsCutCutComp: Index of the selected cut component.
        - resultsCutGridLines: Boolean indicating if grid lines are checked.
        - resultsCutInterp: Boolean indicating if interpolation is checked.
        - resultsCutCutValues: Text value of the cut values input.
        """
        self.resultsCutImgPair = self.resultsUiCut.imgPairIn.currentIndex()
        self.resultsCutSmoothWin = self.resultsUiCut.smoothWinIn.text()
        self.resultsCutComp = self.resultsUiCut.compIn.currentIndex()
        self.resultsCutSmoothOrder = self.resultsUiCut.smoothOrderIn.text()
        self.resultsCutCutComp = self.resultsUiCut.cutCompIn.currentIndex()
        self.resultsCutGridLines = self.resultsUiCut.gridLinesIn.isChecked()
        self.resultsCutInterp = self.resultsUiCut.interpIn.isChecked()
        self.resultsCutCutValues = self.resultsUiCut.cutValIn.text()

    def submitGraph(self):
        """
        Handles the submission of the graph based on selected options.

        This method checks the values of `option1Ind` and `option2Ind` to determine
        which type of graph to plot. It supports plotting contour displacement, 
        contour strain, cut line displacement, and cut line strain graphs.

        If an error occurs during the plotting process, it captures the exception 
        and displays an error message in a popup dialog.

        Raises:
            Exception: If an error occurs during the plotting process, the error 
            message is displayed in a QMessageBox.
        """
        try:
            matplotlib.pyplot.close()
            if self.option1Ind == 1:
                if self.option2Ind == 0:
                    self.plotContourDisp()
                elif self.option2Ind == 1:
                    self.plotContourStrain()
            elif self.option1Ind == 2:
                if self.option2Ind == 0:
                    self.plotCutLineDisp()
                elif self.option2Ind == 1:
                    self.plotCutLineStrain()
        except Exception as e:
            # Capture the standard error and display it in a popup
            error_message = str(e)
            QMessageBox.critical(
                self, "Error", f"An error occurred: {error_message}")

    def plotContourDisp(self):
        """
        Plots a contour displacement map based on the selected parameters from the UI.
        This method retrieves user inputs from the UI, processes them, and uses them to generate
        a displacement contour plot. The plot is then displayed on the application's canvas.
        Parameters:
        None
        Returns:
        None
        UI Elements Used:
        - self.resultsUiCon.compIn: Component index for displacement computation.
        - self.resultsUiCon.alphaIn: Alpha value for the plot.
        - self.resultsUiCon.smoothWinIn: Smoothing window size.
        - self.resultsUiCon.maxValIn: Maximum value for the plot.
        - self.resultsUiCon.minValIn: Minimum value for the plot.
        - self.resultsUiCon.smoothOrderIn: Smoothing order.
        - self.resultsUiCon.imgPairIn: Image pair index.
        Attributes:
        - self.figure: The generated figure for the displacement contour plot.
        - self.canvas: The canvas widget to display the figure.
        - self.lastImagePair: The last image pair index used for plotting.
        Methods Called:
        - self.resultsSelChanged(): Updates the results selection.
        - sdpp.plotDispContour(): Generates the displacement contour plot.
        """
        # Removing any previous graphs
        self.resultsSelChanged()

        # Getting the required parameters
        dispComp = getattr(sdpp.DispComp, sdpp.DispComp._member_names_[
                           self.resultsUiCon.compIn.currentIndex()])
        alpha = float(self.resultsUiCon.alphaIn.text())
        smoothWindow = int(self.resultsUiCon.smoothWinIn.text())
        if self.resultsUiCon.maxValIn.text() == "None" or self.resultsUiCon.maxValIn.text() == "none":
            maxVal = None
        else:
            maxVal = float(self.resultsUiCon.maxValIn.text())
        if self.resultsUiCon.minValIn.text() == "None" or self.resultsUiCon.minValIn.text() == "none":
            minVal = None
        else:
            minVal = float(self.resultsUiCon.minValIn.text())
        smoothOrder = int(self.resultsUiCon.smoothOrderIn.text())
        imgPair = self.lastImagePair - self.resultsUiCon.imgPairIn.currentIndex()

        # Plotting the graph
        self.figure = sdpp.plotDispContour(self._savePath, imgPair=imgPair, dispComp=dispComp,
                                           alpha=alpha, plotImage=True,
                                           showPlot=False, fileName='', smoothWindow=smoothWindow, smoothOrder=smoothOrder, maxValue=maxVal, minValue=minVal, return_fig=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavigationToolbar(self.canvas, self)
        self.resultsUi.tempLayout.insertWidget(3, toolbar)
        self.resultsUi.tempLayout.insertWidget(4, self.canvas)

    def plotCutLineDisp(self):
        """
        Plots the displacement cut line based on the user-selected parameters.
        This method retrieves user inputs from the GUI, processes them, and generates
        a plot of the displacement cut line. The plot is then displayed on the GUI.
        Parameters:
        None
        Returns:
        None
        """
        # Removing any previous graphs
        self.resultsSelChanged()

        # Getting the required parameters
        imgPair = self.lastImagePair - self.resultsUiCut.imgPairIn.currentIndex()
        dispComp = getattr(sdpp.DispComp, sdpp.DispComp._member_names_[
                           self.resultsUiCut.compIn.currentIndex()])
        cutComp = getattr(sdpp.CompID, sdpp.CompID._member_names_[
                          self.resultsUiCut.cutCompIn.currentIndex()])
        cutValues = [int(i)
                     for i in self.resultsUiCut.cutValIn.text().split(",")]
        gridlines = self.resultsUiCut.gridLinesIn.isChecked()
        smoothWindow = int(self.resultsUiCut.smoothWinIn.text())
        smoothOrder = int(self.resultsUiCut.smoothOrderIn.text())
        interpolate = self.resultsUiCut.interpIn.isChecked()

        # Plotting the graph
        self.figure = sdpp.plotDispCutLine(self._savePath, imgPair=imgPair, dispComp=dispComp, cutComp=cutComp,
                                           cutValues=cutValues, gridLines=gridlines, showPlot=False,
                                           fileName='', smoothWindow=smoothWindow, smoothOrder=smoothOrder, interpolate=interpolate, return_fig=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavigationToolbar(self.canvas, self)
        self.resultsUi.tempLayout.insertWidget(3, toolbar)
        self.resultsUi.tempLayout.insertWidget(4, self.canvas)

    def plotContourStrain(self):
        """
        Plots the contour strain based on the user-selected parameters.
        This method retrieves user inputs for strain component, alpha value, smoothing window, 
        maximum and minimum values, smoothing order, and image pair index. It then calls the 
        `plotStrainContour` function from the `sdpp` module to generate the strain contour plot 
        and embeds the resulting plot into the GUI.
        Parameters:
        None
        Returns:
        None
        """
        # Removing any previous graphs
        self.resultsSelChanged()

        # Getting the required parameters
        strainComp = getattr(sdpp.StrainComp, sdpp.StrainComp._member_names_[
                             self.resultsUiCon.compIn.currentIndex()])
        alpha = float(self.resultsUiCon.alphaIn.text())
        smoothWindow = int(self.resultsUiCon.smoothWinIn.text())
        if self.resultsUiCon.maxValIn.text() == "None" or self.resultsUiCon.maxValIn.text() == "none":
            maxVal = None
        else:
            maxVal = float(self.resultsUiCon.maxValIn.text())
        if self.resultsUiCon.minValIn.text() == "None" or self.resultsUiCon.minValIn.text() == "none":
            minVal = None
        else:
            minVal = float(self.resultsUiCon.minValIn.text())
        smoothOrder = int(self.resultsUiCon.smoothOrderIn.text())
        imgPair = self.lastImagePair - self.resultsUiCon.imgPairIn.currentIndex()

        # Plotting the graph
        self.figure = sdpp.plotStrainContour(self._savePath, imgPair=imgPair, strainComp=strainComp,
                                             alpha=alpha, plotImage=True,
                                             showPlot=False, fileName='', smoothWindow=smoothWindow, smoothOrder=smoothOrder, maxValue=maxVal, minValue=minVal, return_fig=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavigationToolbar(self.canvas, self)
        self.resultsUi.tempLayout.insertWidget(3, toolbar)
        self.resultsUi.tempLayout.insertWidget(4, self.canvas)

    def plotCutLineStrain(self):
        """
        Plots the strain cut line based on the user-selected parameters from the GUI.
        This method retrieves the user-selected parameters for the strain cut line plot,
        processes them, and generates the plot using the `sdpp.plotStrainCutLine` function.
        The plot is then displayed on the GUI.
        Parameters:
        None
        Returns:
        None
        """
        # Removing any previous graphs
        self.resultsSelChanged()

        # Getting the required parameters
        imgPair = self.lastImagePair - self.resultsUiCut.imgPairIn.currentIndex()
        strainComp = getattr(sdpp.StrainComp, sdpp.StrainComp._member_names_[
                             self.resultsUiCut.compIn.currentIndex()])
        cutComp = getattr(sdpp.CompID, sdpp.CompID._member_names_[
                          self.resultsUiCut.cutCompIn.currentIndex()])
        cutValues = [int(i)
                     for i in self.resultsUiCut.cutValIn.text().split(",")]
        gridlines = self.resultsUiCut.gridLinesIn.isChecked()
        smoothWindow = int(self.resultsUiCut.smoothWinIn.text())
        smoothOrder = int(self.resultsUiCut.smoothOrderIn.text())
        interpolate = self.resultsUiCut.interpIn.isChecked()

        # Plotting the graph
        self.figure = sdpp.plotStrainCutLine(self._savePath, imgPair=imgPair, strainComp=strainComp, cutComp=cutComp,
                                             cutValues=cutValues, gridLines=gridlines, showPlot=False,
                                             fileName='', smoothWindow=smoothWindow, smoothOrder=smoothOrder, interpolate=interpolate, return_fig=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavigationToolbar(self.canvas, self)
        self.resultsUi.tempLayout.insertWidget(3, toolbar)
        self.resultsUi.tempLayout.insertWidget(4, self.canvas)

    def anew(self):
        """
        Creates a new settings file by resetting all settings to their default values.
        If the current settings have unsaved changes, a warning message is displayed to the user.
        The user is prompted to confirm whether they want to proceed with creating a new settings file.
        If the user confirms, all settings are reset to their default values, and the window title is updated.
        If the user cancels, the method returns without making any changes.
        """
        if self.flag00 == True and self.flag01 == False:
            warn = QMessageBox.question(
                self, "Warning", "You have unsaved changes. Do you want to proceed?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if warn == QMessageBox.No:
                return

        # If the main frame layout exists, delete it. Prevents multiple layouts from being created.
        if hasattr(self, 'mainFrameLayout'):
            if self.mainFrame.layout() is not None:
                self.deleteLayout(self.mainFrameLayout)

        self.flag00 = False
        self.flag01 = False
        self.flag02 = False
        self.flag03 = False

        # Setting default values - NB: Some things need 'translation' as the GUI uses indexes for some items and the settings class uses strings
        self._debugLevel = self._defaultSettings.DebugLevel
        self._imageFolder = self._defaultSettings.ImageFolder
        self._CPUCount = self._defaultSettings.CPUCount
        #
        if self._defaultSettings.DICType == 'Planar':
            self._DICType = 0
        elif self._defaultSettings.DICType == 'Stereo':
            self._DICType = 1
        self._subSetSize = self._defaultSettings.SubsetSize
        self._stepSize = self._defaultSettings.StepSize
        #
        if self._defaultSettings.ShapeFunctions == 'Affine':
            self._shapeFunc = 0
        elif self._defaultSettings.ShapeFunctions == 'Quadratic':
            self._shapeFunc = 1
        self._startingPoints = self._defaultSettings.StartingPoints
        #
        if self._defaultSettings.ReferenceStrategy == 'Relative':
            self._refStrat = 0
        elif self._defaultSettings.ReferenceStrategy == 'Absolute':
            self._refStrat = 1
        self._gausBlurSize = self._defaultSettings.GaussianBlurSize
        self._gausBlurSTD = self._defaultSettings.GaussianBlurStdDev
        self._datumImage = self._defaultSettings.DatumImage + 1
        self._targetImage = self._defaultSettings.TargetImage
        self._increment = self._defaultSettings.Increment
        self._ROI = self._defaultSettings.ROI
        self._backgroundCutOff = self._defaultSettings.BackgroundCutoff
        #
        if self._defaultSettings.OptimizationAlgorithm == 'IC-GN':
            self._optAlgor = 0
        elif self._defaultSettings.OptimizationAlgorithm == 'IC-LM':
            self._optAlgor = 1
        elif self._defaultSettings.OptimizationAlgorithm == 'Fast-IC-LM':
            self._optAlgor = 2

        self._maxIter = self._defaultSettings.MaxIterations
        self._convTol = self._defaultSettings.ConvergenceThreshold
        self._znccTol = self._defaultSettings.NZCCThreshold

        self.setWindowTitle("SUN-DIC")

    def asave(self):
        """
        Saves the current settings to a file. If results already exist for the file, 
        a warning is displayed to the user indicating that saving new settings will 
        delete the previous results. The user is given the option to proceed or cancel.
        The settings saved include:
        - DIC Type (Planar or Stereo)
        - Shape Functions (Affine or Quadratic)
        - Reference Strategy (Relative or Absolute)
        - Optimization Algorithm (IC-GN, IC-LM or Fast-IC-LM)
        - Datum and Target Images
        - Debug Level
        - Subset Size
        - Step Size
        - Starting Points
        - Gaussian Blur Size and Standard Deviation
        - Increment
        - Region of Interest (ROI)
        - Background Cutoff
        - Maximum Iterations
        - Convergence Threshold
        - Image Folder
        - CPU Count
        If the save path is not specified, the user is prompted to provide a save path.
        After saving, the window title is updated to reflect the saved file name.
        Returns:
            None
        """
        if self.flag01 == True and self.flag03 == False:
            warn = QMessageBox.question(self, "Warning", "As results already exist for this file, new settings cannot be saved as this would cause a discrepency between the settings and the results. If you choose to save the new settings, the previous results will be deleted. Do you want to proceed?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if warn == QMessageBox.No:
                return
            else:
                self.flag01 = False
                self.resultsBut.setEnabled(False)

        if self._savePath is None:
            self.asaveAs()
            return

        if self._DICType == 0:
            _DICType = 'Planar'
        elif self._DICType == 1:
            _DICType = 'Stereo'
        if self._shapeFunc == 0:
            _shapeFunc = 'Affine'
        elif self._shapeFunc == 1:
            _shapeFunc = 'Quadratic'
        if self._refStrat == 0:
            _refStrat = 'Relative'
        elif self._refStrat == 1:
            _refStrat = 'Absolute'
        if self._optAlgor == 0:
            _optAlgor = 'IC-GN'
        elif self._optAlgor == 1:
            _optAlgor = 'IC-LM'
        elif self._optAlgor == 2:
            _optAlgor = 'Fast-IC-LM'

        _datumImage = int(self._datumImage) - 1
        if int(self._targetImage) == -1:
            _targetImage = -1
        else:
            _targetImage = int(self._targetImage) - 1

        dicSet = dicSettings()
        dicSet.DebugLevel = int(self._debugLevel)
        dicSet.DICType = _DICType
        dicSet.DICType = _DICType
        dicSet.ShapeFunctions = _shapeFunc
        dicSet.SubsetSize = int(self._subSetSize)
        dicSet.StepSize = int(self._stepSize)
        dicSet.StartingPoints = int(self._startingPoints)
        dicSet.ReferenceStrategy = _refStrat
        dicSet.GaussianBlurSize = int(self._gausBlurSize)
        dicSet.GaussianBlurStdDev = float(self._gausBlurSTD)
        dicSet.DatumImage = _datumImage
        dicSet.TargetImage = _targetImage
        dicSet.Increment = int(self._increment)
        dicSet.ROI = self._ROI
        dicSet.BackgroundCutoff = int(self._backgroundCutOff)
        dicSet.OptimizationAlgorithm = _optAlgor
        dicSet.MaxIterations = int(self._maxIter)
        dicSet.ConvergenceThreshold = float(self._convTol)
        dicSet.NZCCThreshold = float(self._znccTol)
        dicSet.ImageFolder = self._imageFolder
        dicSet.CPUCount = int(self._CPUCount)

        f = df.DataFile.openWriter(self._savePath)
        f.writeHeading(dicSet)
        f.close
        self.flag00 = False
        self.setWindowTitle(
            str(os.path.basename(self._savePath)) + " - SUN-DIC")

    def asaveAs(self):
        """
        Saves the current settings to a file. If results already exist for the current file and new settings are being saved, 
        a warning is displayed to the user indicating that the previous results will be deleted if they proceed.
        The settings saved include:
        - DIC Type (Planar or Stereo)
        - Shape Functions (Affine or Quadratic)
        - Reference Strategy (Relative or Absolute)
        - Optimization Algorithm (IC-GN, IC-LM or Fast-IC-LM)
        - Datum and Target Images
        - Debug Level
        - Subset Size
        - Step Size
        - Starting Points
        - Gaussian Blur Size and Standard Deviation
        - Increment
        - Region of Interest (ROI)
        - Background Cutoff
        - Maximum Iterations
        - Convergence Threshold
        - Image Folder
        - CPU Count
        The user is prompted to choose a file location and name for saving the settings. The settings are saved in a binary 
        file with the extension '.sdic'. The window title is updated to reflect the new file name.
        Raises:
            None
        """

        if self.flag01 == True and self.flag03 == False:
            warn = QMessageBox.question(self, "Warning", "As results already exist for this file, new settings cannot be saved as this would cause a discrepency between the settings and the results. If you choose to save the new settings, the previous results will be deleted. Do you want to proceed?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if warn == QMessageBox.No:
                return
            else:
                self.flag01 = False
                self.resultsBut.setEnabled(False)

        if self._DICType == 0:
            _DICType = 'Planar'
        elif self._DICType == 1:
            _DICType = 'Stereo'
        if self._shapeFunc == 0:
            _shapeFunc = 'Affine'
        elif self._shapeFunc == 1:
            _shapeFunc = 'Quadratic'
        if self._refStrat == 0:
            _refStrat = 'Relative'
        elif self._refStrat == 1:
            _refStrat = 'Absolute'
        if self._optAlgor == 0:
            _optAlgor = 'IC-GN'
        elif self._optAlgor == 1:
            _optAlgor = 'IC-LM'
        elif self._optAlgor == 2:
            _optAlgor = 'Fast-IC-LM'

        _datumImage = int(self._datumImage) - 1
        if int(self._targetImage) == -1:
            _targetImage = -1
        else:
            _targetImage = int(self._targetImage) - 1

        dicSet = dicSettings()
        dicSet.DebugLevel = int(self._debugLevel)
        dicSet.DICType = _DICType
        dicSet.DICType = _DICType
        dicSet.ShapeFunctions = _shapeFunc
        dicSet.SubsetSize = int(self._subSetSize)
        dicSet.StepSize = int(self._stepSize)
        dicSet.StartingPoints = int(self._startingPoints)
        dicSet.ReferenceStrategy = _refStrat
        dicSet.GaussianBlurSize = int(self._gausBlurSize)
        dicSet.GaussianBlurStdDev = float(self._gausBlurSTD)
        dicSet.DatumImage = _datumImage
        dicSet.TargetImage = _targetImage
        dicSet.Increment = int(self._increment)
        dicSet.ROI = self._ROI
        dicSet.BackgroundCutoff = int(self._backgroundCutOff)
        dicSet.OptimizationAlgorithm = _optAlgor
        dicSet.MaxIterations = int(self._maxIter)
        dicSet.ConvergenceThreshold = float(self._convTol)
        dicSet.NZCCThreshold = float(self._znccTol)
        dicSet.ImageFolder = self._imageFolder
        dicSet.CPUCount = int(self._CPUCount)
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,
                                                            "Save File", "", "SUNDIC Binary File(*.sdic)", options=options)
        if fileName:
            if not fileName.endswith('.sdic'):
                newFileName = fileName+'.sdic'
            else:
                newFileName = fileName
            f = df.DataFile.openWriter(newFileName)
            f.writeHeading(dicSet)
            f.close
            self._savePath = newFileName
            self.setWindowTitle(
                str(os.path.basename(newFileName)) + " - SUN-DIC")
            self._savePath = newFileName
            self.flag00 = False

    def aopen(self):
        """
        Opens a file dialog to select a SUNDIC Binary File (*.sdic) and loads its settings.
        This method performs the following steps:
        1. Opens a file dialog to select a file.
        2. If a file is selected, it updates the window title with the file name.
        3. Loads settings from the selected file using the `sdset.Settings.fromMsgPackFile` method.
        4. Updates various instance variables with the loaded settings.
        5. Enables the results button if the file contains results.
        Attributes:
            _savePath (str): Path to the selected file.
            _debugLevel (int): Debug level from the settings.
            _imageFolder (str): Image folder path from the settings.
            _CPUCount (int): Number of CPUs to use from the settings.
            _DICType (int): Type of DIC (0 for Planar, 1 for Stereo).
            _subSetSize (int): Subset size from the settings.
            _stepSize (int): Step size from the settings.
            _shapeFunc (int): Shape function type (0 for Affine, 1 for Quadratic).
            _startingPoints (list): Starting points from the settings.
            _refStrat (int): Reference strategy (0 for Relative, 1 for Absolute).
            _gausBlurSize (int): Gaussian blur size from the settings.
            _gausBlurSTD (float): Gaussian blur standard deviation from the settings.
            _datumImage (int): Datum image index from the settings (1-based).
            _targetImage (int): Target image index from the settings.
            _increment (int): Increment value from the settings.
            _ROI (tuple): Region of interest from the settings.
            _backgroundCutOff (float): Background cutoff value from the settings.
            _optAlgor (int): Optimization algorithm (0 for IC-GN, 1 for IC-LM, 2 for Fast-IC-LM).
            _maxIter (int): Maximum number of iterations from the settings.
            _convTol (float): Convergence threshold from the settings.
            _znccTol (float): ZNCC tolerance from the settings. 
            flag01 (bool): Flag indicating if the file contains results.
            flag02 (bool): Flag for ROI drawer.
        """
        if self.flag00 == True and self.flag01 == False:
            warn = QMessageBox.question(
                self, "Warning", "You have unsaved changes. Do you want to proceed?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if warn == QMessageBox.No:
                return

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "SUNDIC Binary Files (*.sdic);;All Files (*)", options=options)

        if file_path:
            if hasattr(self, 'mainFrameLayout'):
                if self.mainFrame.layout() is not None:
                    self.deleteLayout(self.mainFrameLayout)
            self.setWindowTitle(
                str(os.path.basename(file_path)) + " - SUN-DIC")
            dicSet = sdset.Settings.fromMsgPackFile(file_path)
            self._savePath = file_path
            self._debugLevel = dicSet.DebugLevel
            self._imageFolder = dicSet.ImageFolder
            self._CPUCount = dicSet.CPUCount
            #
            if dicSet.DICType == 'Planar':
                self._DICType = 0
            elif dicSet.DICType == 'Stereo':
                self._DICType = 1
            self._subSetSize = dicSet.SubsetSize
            self._stepSize = dicSet.StepSize
            #
            if dicSet.ShapeFunctions == 'Affine':
                self._shapeFunc = 0
            elif dicSet.ShapeFunctions == 'Quadratic':
                self._shapeFunc = 1
            self._startingPoints = dicSet.StartingPoints
            #
            if dicSet.ReferenceStrategy == 'Relative':
                self._refStrat = 0
            elif dicSet.ReferenceStrategy == 'Absolute':
                self._refStrat = 1
            self._gausBlurSize = dicSet.GaussianBlurSize
            self._gausBlurSTD = dicSet.GaussianBlurStdDev
            self._datumImage = dicSet.DatumImage + 1
            self._targetImage = dicSet.TargetImage
            self._increment = dicSet.Increment
            self._ROI = dicSet.ROI
            self._backgroundCutOff = dicSet.BackgroundCutoff
            #
            if dicSet.OptimizationAlgorithm == 'IC-GN':
                self._optAlgor = 0
            elif dicSet.OptimizationAlgorithm == 'IC-LM':
                self._optAlgor = 1
            elif dicSet.OptimizationAlgorithm == 'Fast-IC-LM':
                self._optAlgor = 2

            self._maxIter = dicSet.MaxIterations
            self._convTol = dicSet.ConvergenceThreshold
            self._znccTol = dicSet.NZCCThreshold

            self.flag02 = True

            try:
                temp = df.DataFile.openReader(self._savePath)
                temp.readSubSetData(0)
                self.resultsBut.setEnabled(True)
                self.flag01 = True
                temp.close()
            except:
                self.flag01 = False
                temp.close()

    def aexit(self):
        """
        Closes the application window.
        """
        self.close()

    def agithub(self):
        webbrowser.open_new_tab('https://github.com/gventer/SUN-DIC')

    def aversion(self):
        version = sdversion.__version__
        QMessageBox.about(self, "Version", f"SUN-DIC Version: {version}")

    def setDefaultsSettings(self):
        """
        Sets the default settings for the Settings page based on the values from the _defaultSettings attribute.
        This method translates certain string settings into corresponding index values used by the GUI components.
        It updates the GUI elements with these default values and sets a flag to indicate that there are unsaved changes.
        Attributes:
            _defaultSettings (object): An object containing the default settings.
            _DICType (int): Index for DIC type (0 for 'Planar', 1 for 'Stereo').
            _subSetSize (int): Subset size for DIC analysis.
            _stepSize (int): Step size for DIC analysis.
            _shapeFunc (int): Index for shape functions (0 for 'Affine', 1 for 'Quadratic').
            _startingPoints (int): Number of starting points for the analysis.
            _refStrat (int): Index for reference strategy (0 for 'Relative', 1 for 'Absolute').
            _optAlgor (int): Index for optimization algorithm (0 for 'IC-GN', 1 for 'IC-LM', 2 for 'Fast-IC-LM').
            _maxIter (int): Maximum number of iterations for the optimization algorithm.
            _convTol (float): Convergence threshold for the optimization algorithm.
            _znccTol (float): ZNCC threshold for the analysis.
            settingsUI (object): The UI object containing the GUI elements to be updated.
            flag00 (bool): A flag indicating that there are unsaved changes.
        Updates:
            settingsUI.subsetSizeIn: Sets the text for the subset size input field.
            settingsUI.stepSizeIn: Sets the text for the step size input field.
            settingsUI.startingPIn: Sets the text for the starting points input field.
            settingsUI.maxItIn: Sets the text for the maximum iterations input field.
            settingsUI.dicTypeBox: Sets the current index for the DIC type combo box.
            settingsUI.shapeFuncBox: Sets the current index for the shape functions combo box.
            settingsUI.convergenceIn: Sets the text for the convergence threshold input field.
            settingsUI.refBox: Sets the current index for the reference strategy combo box.
            settingsUI.algoTypeBox: Sets the current index for the optimization algorithm combo box.
        """
        # Setting default values - NB: Some things need 'translation' as the GUI uses indexes for some items and the settings class uses strings
        if self._defaultSettings.DICType == 'Planar':
            self._DICType = 0
        elif self._defaultSettings.DICType == 'Stereo':
            self._DICType = 1
        self._subSetSize = self._defaultSettings.SubsetSize
        self._stepSize = self._defaultSettings.StepSize
        #
        if self._defaultSettings.ShapeFunctions == 'Affine':
            self._shapeFunc = 0
        elif self._defaultSettings.ShapeFunctions == 'Quadratic':
            self._shapeFunc = 1
        self._startingPoints = self._defaultSettings.StartingPoints
        #
        if self._defaultSettings.ReferenceStrategy == 'Relative':
            self._refStrat = 0
        elif self._defaultSettings.ReferenceStrategy == 'Absolute':
            self._refStrat = 1
        #
        if self._defaultSettings.OptimizationAlgorithm == 'IC-GN':
            self._optAlgor = 0
        elif self._defaultSettings.OptimizationAlgorithm == 'IC-LM':
            self._optAlgor = 1
        elif self._defaultSettings.OptimizationAlgorithm == 'Fast-IC-LM':
            self._optAlgor = 2

        self._maxIter = self._defaultSettings.MaxIterations
        self._convTol = self._defaultSettings.ConvergenceThreshold
        self._znccTol = self._defaultSettings.NZCCThreshold

        self.settingsUI.subsetSizeIn.setText(str(self._subSetSize))
        self.settingsUI.stepSizeIn.setText(str(self._stepSize))
        self.settingsUI.startingPIn.setText(str(self._startingPoints))
        self.settingsUI.maxItIn.setText(str(self._maxIter))
        self.settingsUI.dicTypeBox.setCurrentIndex(self._DICType)
        self.settingsUI.shapeFuncBox.setCurrentIndex(self._shapeFunc)
        self.settingsUI.convergenceIn.setText(str(self._convTol))
        self.settingsUI.znccTolIn.setText(str(self._znccTol))
        self.settingsUI.refBox.setCurrentIndex(self._refStrat)
        self.settingsUI.algoTypeBox.setCurrentIndex(self._optAlgor)
        self.flag00 = True
        self.showUnsaved()

    def setDefaultsImageSet(self):
        """
        Sets the default values for the image set configuration and updates the UI accordingly.
        This method initializes various settings related to image processing from the default settings
        and updates the corresponding UI elements to reflect these values. It also sets the default
        image folder to the user's Pictures directory, retrieves the filenames of all images in this
        directory, sorts them naturally, and populates the UI with these filenames.
        Attributes:
            self._imageFolder (str): Path to the image folder.
            self._gausBlurSize (int): Size of the Gaussian blur.
            self._gausBlurSTD (float): Standard deviation of the Gaussian blur.
            self._datumImage (int): Index of the datum image.
            self._targetImage (int): Index of the target image.
            self._increment (int): Increment value for image processing.
            self._backgroundCutOff (float): Background cutoff value.
        UI Elements Updated:
            self.imageSetUi.startIn (QLineEdit): Start image index input field.
            self.imageSetUi.endIn (QLineEdit): End image index input field.
            self.imageSetUi.incIn (QLineEdit): Increment input field.
            self.imageSetUi.gausIn (QLineEdit): Gaussian blur size input field.
            self.imageSetUi.backIn (QLineEdit): Background cutoff input field.
            self.imageSetUi.folderDisp (QLineEdit): Folder display field.
            self.imageSetUi.dispImages (QListView): List view to display image filenames.
        Flags:
            self.flag00 (bool): A flag indicating that the default settings have been applied.
        Methods Called:
            self.showUnsaved(): Indicates that there are unsaved changes.
        """

        # Setting default values
        self._imageFolder = self._defaultSettings.ImageFolder
        self._gausBlurSize = self._defaultSettings.GaussianBlurSize
        self._gausBlurSTD = self._defaultSettings.GaussianBlurStdDev
        self._datumImage = self._defaultSettings.DatumImage + 1
        self._targetImage = self._defaultSettings.TargetImage
        self._increment = self._defaultSettings.Increment
        self._backgroundCutOff = self._defaultSettings.BackgroundCutoff
        #

        self.imageSetUi.folderDisp.setText(QStandardPaths.standardLocations(
            QStandardPaths.PicturesLocation)[0])
        self._imageFolder = QStandardPaths.standardLocations(
            QStandardPaths.PicturesLocation)[0]
        # Get the filenames of all images in the directory
        files = os.listdir(self._imageFolder)
        # Do a natural sort on the filenames
        files = ns.os_sorted(files)
        self._targetImage = len(files)
        self.imageSetUi.model = QStandardItemModel()
        self.imageSetUi.dispImages.setModel(self.imageSetUi.model)

        for i in files:
            item = QStandardItem(i)
            self.imageSetUi.model.appendRow(item)

        self.imageSetUi.startIn.setText(str(self._datumImage))
        self.imageSetUi.endIn.setText(str(self._targetImage))
        self.imageSetUi.incIn.setText(str(self._increment))
        self.imageSetUi.gausIn.setText(str(self._gausBlurSize))
        self.imageSetUi.backIn.setText(str(self._backgroundCutOff))
        self.imageSetUi.folderDisp.setText(self._imageFolder)
        self.flag00 = True
        self.showUnsaved()

    def deleteLayout(self, layout):
        """
        Deletes all widgets and nested layouts from the given layout.

        This method iterates through all items in the provided layout, removing
        and deleting each widget or nested layout it encounters. If an item is a 
        widget, it is scheduled for deletion using `deleteLater()`. If an item is 
        a nested layout, the method calls itself recursively to delete all items 
        within the nested layout. Finally, the layout itself is deleted.

        Args:
            layout (QLayout): The layout to be deleted.
        """
        # Method to delete the layout
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.deleteLayout(item.layout())
            sip.delete(layout)

    def handleCoords(self, point):
        """
        Updates the label with the coordinates of the given point if it is not null.

        Args:
            point (QPoint): The point whose coordinates are to be displayed.

        If the point is not null, its x and y coordinates are set as the text of the labelCoords.
        If the point is null, the labelCoords is cleared.
        """
        if not point.isNull():
            self.labelCoords.setText(f'{point.x()}, {point.y()}')
        else:
            self.labelCoords.clear()

    def handleOpen(self):
        """
        Handles the action of opening an image file.

        This method opens a file dialog for the user to select an image file. 
        If a file is selected and successfully loaded, it sets the image to 
        the viewer and updates the path label. If the image cannot be loaded, 
        it shows an error message.

        Steps:
        1. Determine the starting directory for the file dialog.
        2. Open the file dialog to select an image file.
        3. Clear any existing coordinates on the label.
        4. If the selected file is a valid image, set it to the viewer and 
           update the internal path.
        5. If the image is invalid, display an error message.
        6. Update the viewer path label with the current path.

        Attributes:
            self._path (str): The current path of the image file.
            self.labelCoords (QLabel): The label displaying coordinates.
            self.viewer (QGraphicsView): The viewer displaying the image.
            self.viewerPathLabel (QLabel): The label displaying the path of the image file.
        """
        if (start := self._path) is None:
            start = QStandardPaths.standardLocations(
                QStandardPaths.PicturesLocation)[0]
        if path := QFileDialog.getOpenFileName(
                self, 'Open Image', start)[0]:
            self.labelCoords.clear()
            if not (pixmap := QPixmap(path)).isNull():
                self.viewer.setPhoto(pixmap)
                self._path = path
            else:
                QMessageBox.warning(self, 'Error',
                                    f'<br>Could not load image file:<br>'
                                    f'<br><b>{path}</b><br>'
                                    )
        self.viewerPathLabel.setText(self._path)

    def openImageSetFolder(self):
        """
        Opens a dialog to select a directory containing image sets, updates the UI with the selected directory,
        and populates the image list display with the images from the selected directory based on the specified
        range and increment.

        This method performs the following steps:
        1. Opens a QFileDialog to select a directory.
        2. Updates the folder display in the UI with the selected directory.
        3. Lists the files in the selected directory and sorts them.
        4. Initializes a QStandardItemModel and sets it to the image display view.
        5. Filters the files based on the specified start, end, and increment values.
        6. Populates the image display view with the filtered files.
        7. Calls the changedImageSet method to handle any additional updates.

        Attributes:
            self.imageSetUi.folderDisp (QLabel): UI element to display the selected folder path.
            self._imageFolder (str): Path to the selected image folder.
            self.imageSetUi.model (QStandardItemModel): Model to hold the list of images.
            self.imageSetUi.dispImages (QListView): UI element to display the list of images.
            self._datumImage (int): Starting index for the image range.
            self._targetImage (int): Ending index for the image range.
            self._increment (int): Increment value for selecting images within the range.

        """
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", "", options=options)
        if directory:
            self.imageSetUi.folderDisp.setText(directory)
            self._imageFolder = directory
            files = os.listdir(self._imageFolder)
            files = ns.os_sorted(files)
            self._targetImage = len(files)
            self.imageSetUi.model = QStandardItemModel()
            self.imageSetUi.dispImages.setModel(self.imageSetUi.model)
            start = int(self._datumImage) - 1
            end = int(self._targetImage)
            inc = int(self._increment)
            files = files[start:end:inc]
            for i in files:
                item = QStandardItem(i)
                self.imageSetUi.model.appendRow(item)
            self.imageSetUi.endIn.setText(str(self._targetImage))
            self.changedImageSet()

    def run_planarDICLocal(self, settings, results_file):
        """
        Runs the Planar Digital Image Correlation (DIC) process locally using the provided settings and results file.

        Args:
            settings (dict): A dictionary containing the settings for the DIC process.
            results_file (str): The path to the file where the results will be saved.

        Initializes a PlanarDICWorker with the given settings and results file, connects the worker's signals to the appropriate
        slots for updating progress, handling completion, and starting the process, and then starts the worker.
        """
        self.worker = PlanarDICWorker(settings, results_file)
        self.worker.progress.connect(self.update_progOut)
        self.worker.finished.connect(self.finished_progOut)
        self.worker.started.connect(self.started_progOut)
        self.worker.start()

    def started_progOut(self, text):
        """
        Updates the status label in the analysis UI with the provided text.

        Args:
            text (str): The text to set as the status.
        """
        self.analysisUi.statusLab.setText("Status: "+text)
        self.analysisUi.statusLab.setStyleSheet("color: green")

    def finished_progOut(self, text):
        """
        Re-enables buttons and updates status label when the program finishes processing.

        Args:
            text (str): The status message to be displayed.
        """
        self.settingsBut.setEnabled(True)
        self.imageSetBut.setEnabled(True)
        self.roiBut.setEnabled(True)
        self.analysisBut.setEnabled(True)
        self.resultsBut.setEnabled(True)
        self.analysisUi.startBut.setEnabled(True)
        self.analysisUi.statusLab.setText("Status: "+text)
        self.analysisUi.statusLab.setStyleSheet("color: green")

    def update_progOut(self, text):
        """
        Updates the progress output display with the given text.

        Args:
            text (str): The text to append to the progress output display.
        """
        if text.strip():
            self.analysisUi.progOut.append(text)

    def showUnsaved(self):
        """
        Updates the window title to indicate unsaved changes.

        If the flag01 attribute is set to True, the method returns immediately 
        without making any changes, this is done because results already
        exist for the file. If the _savePath attribute is None, the 
        window title is set to "* SUN-DIC". Otherwise, the window title is set 
        to the basename of the _savePath followed by "* - SUN-DIC".

        Returns:
            None
        """
        if self._savePath is None:
            self.setWindowTitle("* SUN-DIC")
        else:
            self.setWindowTitle(
                str(os.path.basename(self._savePath)) + "* - SUN-DIC")

    def closeEvent(self, event):
        """
        Handles the close event of the window.
        This method is called when the user attempts to close the window. It checks the state of certain flags and prompts the user with a message box to confirm the action if necessary.
        Parameters:
        event (QCloseEvent): The close event triggered by the user.
        Behavior:
        - If `flag00` is True and `flag01` is False, a message box is shown asking the user if they want to exit without saving.
            - If the user selects 'Yes', the event is accepted and the window closes.
            - If the user selects 'Save', the `asave` method is called to save the data. If `flag00` is then set to False, the event is accepted and the window closes; otherwise, the event is ignored.
            - If the user selects 'No', the event is ignored and the window remains open.
        """
        if self.flag00 == True:
            reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to exit without saving?',
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Save, QMessageBox.No)

            if reply == QMessageBox.Yes:
                event.accept()
            elif reply == QMessageBox.Save:
                self.asave()
                if self.flag00 == False:
                    event.accept()
                else:
                    event.ignore()
            else:
                event.ignore()


class dicSettings:
    def __init__(self):
        self.DebugLevel = None
        self.ImageFolder = None
        self.CPUCount = None
        self.DICType = None
        self.SubsetSize = None
        self.StepSize = None
        self.ShapeFunctions = None
        self.StartingPoints = None
        self.ReferenceStrategy = None
        self.GaussianBlurSize = None
        self.GaussianBlurStdDev = None
        self.DatumImage = None
        self.TargetImage = None
        self.Increment = None
        self.ROI = None
        self.BackgroundCutoff = None
        self.OptimizationAlgorithm = None
        self.MaxIterations = None
        self.ConvergenceThreshold = None


class PlanarDICWorker(QThread):
    """
    A QThread subclass to handle the execution of Planar Digital Image Correlation (DIC) in a separate thread.
    Signals:
        progress (str): Emitted periodically with the current output of the DIC process.
        finished (str): Emitted when the DIC process has finished.
        started (str): Emitted when the DIC process has started.
    Args:
        settings (dict): Configuration settings for the DIC process.
        results_file (str): Path to the file where results will be saved.
    Methods:
        run():
            Overrides the QThread run method to execute the DIC process in a separate thread.
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    started = pyqtSignal(str)

    def __init__(self, settings, results_file):
        super().__init__()
        self.settings = settings
        self.results_file = results_file
        self._stop_event = threading.Event()

    def run(self):
        old_stdout = sys.stdout
        f = io.StringIO()
        sys.stdout = f

        def run_dic():
            try:
                print("RUNNING")
                self.started.emit("RUNNING")
                # subprocess.run(['ray', 'start', '--head', '--port=6379', f'--num-cpus={self.settings.CPUCount}'])
                sd.planarDICLocal(self.settings, self.results_file)
                print("FINISHED")
                self.finished.emit("FINISHED")
                # subprocess.run(['ray', 'stop', '--force'])
            except Exception as e:
                print(f"Exception in thread: {e}")
            finally:
                self._stop_event.set()
                sd.safe_ray_shutdown(externalRay=False)

        dic_thread = threading.Thread(target=run_dic)
        dic_thread.start()

        while not self._stop_event.is_set():
            time.sleep(0.1)
            output = f.getvalue()
            self.progress.emit(output)
            f.truncate(0)
            f.seek(0)

        sys.stdout = old_stdout
        output = f.getvalue()
        self.progress.emit(output)


def main():
    # Make an object of the class and execute it
    app = QApplication(sys.argv)

    # Make an object and call the functions
    win = mainProgram()

    win.show()

    # Exit the window cleanly
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

import sundic.post_process as sdpp
import sundic.util.datafile as dataFile
from sundic.gui.validators import OddNumberValidator, ClampingIntValidator
from sundic.gui.validators import ClampingDblValidator
import os
import numpy as np
import pandas as pd

from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QCheckBox, QSpacerItem, QSizePolicy, QTextEdit, QFileDialog,
    QMessageBox, QStackedLayout, QLayout
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
import matplotlib
matplotlib.use('Qt5Agg')


# ------------------------------------------------------------------------------
# Function to show the graph in the UI - used for both contour and line cut graphs
def showGraph(parent, figure, layout):

    canvas = FigureCanvasQTAgg(figure)
    canvas.setObjectName("canvas")
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                         QSizePolicy.Policy.Expanding)
    toolbar = NavigationToolbar(canvas, parent)
    toolbar.setObjectName("toolbar")

    # Remove old toolbar and canvas if they exist
    widgetNames = ["toolbar", "canvas"]
    for i in range(layout.count(), 0, -1):
        try:
            item = layout.itemAt(i)
            widget = item.widget()
            if widget is not None and widget.objectName() in widgetNames:
                widget.setParent(None)
                widget.deleteLater()
        except Exception as e:
            pass

    # Add the new ones
    layout.addWidget(toolbar)
    layout.addWidget(canvas)


class ResultsUI(QWidget):
    """ Class for the results UI: Defines the layout and widgets for the
        results tab
    """

    # Number of image pairs
    numImagePairs = 0

    # The different UIs for the results tab - they displayed in a stacked layout
    resultsUIExport = None
    resultsUIContour = None
    resultsUILineCut = None

    # The different tab indices for the stacked layout
    EXPRT_TAB = 0
    CONTOUR_TAB = 1
    CUTLINE_TAB = 2

    # ------------------------------------------------------------------------------
    # Initialize the results UI
    def __init__(self, parent):

        super().__init__(parent)

        # Set the class variables
        self.parent = parent

        # Set the number of image pairs
        self.updateImagePairs()
        self.resultsUIExport = ResultsUIExport(self)
        self.resultsUIContour = ResultsUIContour(self)
        self.resultsUILineCut = ResultsUILineCut(self)

        # Define a custom button style
        buttonStyle = "QPushButton{border: 2px  solid  rgb(0, 0, 0);\n" +\
            "background: rgb(255, 255, 255);\n" +\
            "border-style: outset;\n" + \
            "border-width: 1px 1px 1px 1px;\n" + \
            "border-radius: 0px;\n" + \
            "color: black;\n" + \
            "padding: 5px 5px 5px 5px;\n" + \
            "} \n" + \
            "QPushButton:checked {\n" + \
            "border-style: inset;\n" + \
            "background: qradialgradient(\n" + \
            "cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,\n" + \
            "radius: 1.35, stop: 0 #fff, stop: 1 #90D5FF\n" + \
            ");\n" + \
            "}\n" + \
            "QPushButton:disabled {\n" + \
            "border-style: outset;\n" + \
            "background: qradialgradient(\n" + \
            "cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,\n" + \
            "radius: 1.35, stop: 0 #fff, stop: 1 #ddd\n" + \
            ");\n" + \
            "}\n" + \
            ""

        mainVLayout = QVBoxLayout(self)
        mainVLayout.setContentsMargins(10, 10, 10, 10)
        mainVLayout.setSpacing(10)

        horizontalLayout = QHBoxLayout()
        horizontalLayout.setSpacing(0)

        # The text output button
        self.textBut = QPushButton(self)
        self.textBut.setText("Text Output")
        self.textBut.setStyleSheet(buttonStyle)
        self.textBut.setCheckable(True)
        self.textBut.setAutoExclusive(True)
        horizontalLayout.addWidget(self.textBut)

        # The contour graph button
        self.contBut = QPushButton(self)
        self.contBut.setText("Contour Graph")
        self.contBut.setStyleSheet(buttonStyle)
        self.contBut.setCheckable(True)
        self.contBut.setAutoExclusive(True)
        horizontalLayout.addWidget(self.contBut)

        # The line cut graph button
        self.lineBut = QPushButton(self)
        self.lineBut.setText("Line Cut Graph")
        self.lineBut.setStyleSheet(buttonStyle)
        self.lineBut.setCheckable(True)
        self.lineBut.setAutoExclusive(True)
        horizontalLayout.addWidget(self.lineBut)

        spacerItem = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        horizontalLayout.addItem(spacerItem)
        mainVLayout.addLayout(horizontalLayout)

        tempLayout = QVBoxLayout()
        mainVLayout.addLayout(tempLayout)

        # Setup the stacked layout to switch between the different UIs
        self.stackedLayout = QStackedLayout()
        self.stackedLayout.addWidget(self.resultsUIExport)
        self.stackedLayout.addWidget(self.resultsUIContour)
        self.stackedLayout.addWidget(self.resultsUILineCut)
        mainVLayout.addLayout(self.stackedLayout)

        # Setup the default tab view here
        self.stackedLayout.setCurrentIndex(self.EXPRT_TAB)
        self.textBut.setChecked(True)

        # Add connections here
        self.textBut.clicked.connect(self.showTextUI)
        self.contBut.clicked.connect(self.showContUI)
        self.lineBut.clicked.connect(self.showLineUI)

    # ------------------------------------------------------------------------------
    # Function to make the export summary UI visible
    def showTextUI(self):
        self.stackedLayout.setCurrentIndex(self.EXPRT_TAB)

    # ------------------------------------------------------------------------------
    # Function to make the contour graph UI visible
    def showContUI(self):
        self.stackedLayout.setCurrentIndex(self.CONTOUR_TAB)

    # ------------------------------------------------------------------------------
    # Function to make the line cut graph UI visible
    def showLineUI(self):
        self.stackedLayout.setCurrentIndex(self.CUTLINE_TAB)

    # ------------------------------------------------------------------------------
    # Function to get the data from this class and store it in the class variables
    def updateImagePairs(self):
        if self.parent.savePath is not None and os.path.isfile(self.parent.savePath):
            self.numImagePairs = dataFile.DataFile.openReader(
                self.parent.savePath).getNumImagePairs()
        else:
            self.numImagePairs = 0

        return self.numImagePairs


class ResultsUIExport(QWidget):
    """ Class for the results summary UI: Defines the layout and widgets for the
        results summary tab
    """

    # These are the default values shown in this tab
    resultsSumImgPair = 0
    resultsSumSmoothWindow = 3
    resultsSumSmoothOrder = 2
    resultsSumRemoveNan = True
    resultsSumIncDisp = True
    resultsSumIncStrain = True

    # ------------------------------------------------------------------------------
    # Initialize the results summary UI
    def __init__(self, parent):

        super().__init__(parent)
        self.parent = parent

        verticalLayout_2 = QVBoxLayout(self)
        verticalLayout_2.setSpacing(10)

        gridLayout = QGridLayout()
        gridLayout.setHorizontalSpacing(10)
        gridLayout.setVerticalSpacing(10)

        # The image pair input and label
        imgPairLab = QLabel(self)
        imgPairLab.setText("Image Pair:")
        gridLayout.addWidget(imgPairLab, 0, 0, 1, 1)

        self.imgPairIn = QComboBox(self)
        numPairs = self.parent.numImagePairs
        for i in range(numPairs, 0, -1):
            self.imgPairIn.addItem(f"Image Pair {i}")
        gridLayout.addWidget(self.imgPairIn, 0, 1, 1, 1)

        # The smoothing window size input and label
        smoothWindowLab = QLabel(self)
        smoothWindowLab.setText("Smoothing Window:")
        gridLayout.addWidget(smoothWindowLab, 1, 0, 1, 1)

        self.smoothWindowIn = QLineEdit(self)
        self.smoothWindowIn.setText(str(self.resultsSumSmoothWindow))
        smoothWindowValidator = OddNumberValidator(0, None)
        self.smoothWindowIn.setValidator(smoothWindowValidator)
        self.smoothWindowIn.setToolTip("""The size of the window for smoothing the results. 
Must be an odd number.
If only exporting displacement data, then this value can be 0.
If this value is greater than 0, then it must be larger than the smooth order.""")
        gridLayout.addWidget(self.smoothWindowIn, 1, 1, 1, 1)

        # The include displacements checkbox and label
        incDispLab = QLabel(self)
        incDispLab.setText("Include Displacements:")
        gridLayout.addWidget(incDispLab, 2, 0, 1, 1)

        self.incDispIn = QCheckBox(self)
        self.incDispIn.setChecked(bool(self.resultsSumIncDisp))
        gridLayout.addWidget(self.incDispIn, 2, 1, 1, 1)

        # The remove NaN's checkbox and label
        removeNanLab = QLabel(self)
        removeNanLab.setText("Remove NaN\'s:")
        gridLayout.addWidget(removeNanLab, 0, 2, 1, 1)

        self.removeNanIn = QCheckBox(self)
        self.removeNanIn.setChecked(bool(self.resultsSumRemoveNan))
        gridLayout.addWidget(self.removeNanIn, 0, 3, 1, 1)

        verticalLayout_2.addLayout(gridLayout)

        # The smoothing order input and label
        smoothOrderLab = QLabel(self)
        smoothOrderLab.setText("Smoothing Order:")
        gridLayout.addWidget(smoothOrderLab, 1, 2, 1, 1)

        self.smoothOrderIn = QLineEdit(self)
        self.smoothOrderIn.setText(str(self.resultsSumSmoothOrder))
        smoothOrderValidator = ClampingIntValidator()
        smoothOrderValidator.setBottom(0)
        self.smoothOrderIn.setValidator(smoothOrderValidator)
        self.smoothOrderIn.setToolTip(
            "Order of the Savitzky-Golay smoothing polynomial.")
        gridLayout.addWidget(self.smoothOrderIn, 1, 3, 1, 1)

        # The include strains checkbox and label
        incStrainsLab = QLabel(self)
        incStrainsLab.setText("Include Strains:")
        gridLayout.addWidget(incStrainsLab, 2, 2, 1, 1)

        self.incStrainsIn = QCheckBox(self)
        self.incStrainsIn.setChecked(bool(self.resultsSumIncStrain))
        gridLayout.addWidget(self.incStrainsIn, 2, 3, 1, 1)

        # The write data button
        self.writeDataBut = QPushButton(self)
        self.writeDataBut.setText("Export Data")
        verticalLayout_2.addWidget(
            self.writeDataBut, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)

        spacerItem = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        verticalLayout_2.addItem(spacerItem)

        # Add connections here
        self.incDispIn.toggled.connect(self.resultsSumChanged)
        self.incStrainsIn.toggled.connect(self.resultsSumChanged)

        # Connecting the writeDataBut button to the exportData method
        self.writeDataBut.clicked.connect(self.exportData)

    # ------------------------------------------------------------------------------
    # Function that is called to indicate that the data was changed by the user
    def resultsSumChanged(self):
        if self.incDispIn.isChecked() or self.incStrainsIn.isChecked():
            self.writeDataBut.setEnabled(True)
        else:
            self.writeDataBut.setEnabled(False)

    # ------------------------------------------------------------------------------
    # Function to export the data to a CSV file
    def exportData(self):
        try:
            # Saving the settings in correct data format
            imgPair = self.parent.numImagePairs - self.imgPairIn.currentIndex() - 1
            smoothWindow = int(self.resultsSumSmoothWindow)
            smoothOrder = int(self.resultsSumSmoothOrder)
            removeNan = self.resultsSumRemoveNan
            incDisp = self.resultsSumIncDisp
            incStrain = self.resultsSumIncStrain

            # Getting required data
            dispResults, _, _ = sdpp.getDisplacements(
                self.parent.parent.savePath, imgPair=imgPair, smoothWindow=smoothWindow, smoothOrder=smoothOrder)
            strainResults, _, _ = sdpp.getStrains(
                self.parent.parent.savePath, imgPair=imgPair, smoothWindow=smoothWindow, smoothOrder=smoothOrder)

            # Optional remove NaN
            if self.removeNanIn.isChecked():
                dispResults = dispResults[~np.isnan(dispResults).any(axis=1)]
                strainResults = strainResults[~np.isnan(
                    strainResults).any(axis=1)]

            # Saving the data as data frames
            dispDataFrame = pd.DataFrame(dispResults, columns=[
                "X Coord", "Y Coord", "Z Coord", "X Disp", "Y Disp", "Z Disp", "Disp Magnitude"])
            strainDataFrame = pd.DataFrame(strainResults, columns=[
                "X Coord", "Y Coord", "Z Coord", "X Strain Comp", "Y Strain Comp", "XY Strain Comp", "Von Mises Strain"])

            if self.incDispIn.isChecked() and self.incStrainsIn.isChecked():
                results = pd.merge(dispDataFrame, strainDataFrame, how="right", on=[
                                   "X Coord", "Y Coord", "Z Coord"])
            elif self.incDispIn.isChecked():
                results = dispDataFrame
            elif self.incStrainsIn.isChecked():
                results = strainDataFrame

            # Saving the data to a CSV file
            savePath, _ = QFileDialog.getSaveFileName(
                self, "Save File", "", "CSV Files (*.csv)")
            if savePath:
                if not savePath.endswith(".csv"):
                    savePath = savePath + ".csv"
                results.to_csv(savePath, index=False)
        except Exception as e:
            # Capture the standard error and display it in a popup
            error_message = str(e)
            QMessageBox.critical(
                self, "Error", f"An error occurred: {error_message}")


class ResultsUIContour(QWidget):
    """ Class for the results contourplot UI: Defines the layout and widgets for the
        results contourplot tab
    """

    # Define all the variables to store the user input
    resultsType = 0
    resultsComp = 0
    resultsImgPair = 0
    resultsSmoothWindow = 3
    resultsSmoothOrder = 2
    resultsAlpha = 0.75

    DISP_INDEX = 0
    STRAIN_INDEX = 1
    CORELLATION_INDEX = 2

    # ------------------------------------------------------------------------------
    # Function to initialize the UI
    def __init__(self, parent):

        super().__init__(parent)
        self.parent = parent

        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setSpacing(10)

        # Add the results selector combo box
        self.resultsSelector = QComboBox(self)
        boxItems = ["Displacement", "Strain", "Correlation"]
        for item in boxItems:
            self.resultsSelector.addItem(item)
        self.resultsSelector.setCurrentIndex(self.resultsType)
        self.verticalLayout.addWidget(
            self.resultsSelector, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        # Now the grid layout
        self.gridLayout = QGridLayout()
        self.gridLayout.setHorizontalSpacing(10)
        self.gridLayout.setVerticalSpacing(10)

        # The image pair input and label
        self.imgPairLab = QLabel(self)
        self.imgPairLab.setText("Image Pair:")
        self.gridLayout.addWidget(self.imgPairLab, 0, 0, 1, 1)

        self.imgPairIn = QComboBox(self)
        numPairs = self.parent.numImagePairs
        for i in range(numPairs, 0, -1):
            self.imgPairIn.addItem(f"Image Pair {i}")
        self.gridLayout.addWidget(self.imgPairIn, 0, 1, 1, 1)

        # The component input and label
        self.compLab = QLabel(self)
        self.compLab.setText("Component:")
        self.gridLayout.addWidget(self.compLab, 0, 2, 1, 1)

        self.compIn = QComboBox(self)
        for index, e in enumerate(sdpp.DispComp):
            self.compIn.addItem(f"{e.display_name}")
        self.compIn.setCurrentIndex(self.resultsComp)
        self.gridLayout.addWidget(self.compIn, 0, 3, 1, 1)

        # The smoothing window size input and label
        self.smoothWinLab = QLabel(self)
        self.smoothWinLab.setText("Smoothing Window:")
        self.gridLayout.addWidget(self.smoothWinLab, 1, 0, 1, 1)

        self.smoothWinIn = QLineEdit(self)
        self.smoothWinIn.setText(str(self.resultsSmoothWindow))
        smoothWindowValidator = OddNumberValidator(0, None)
        self.smoothWinIn.setValidator(smoothWindowValidator)
        self.smoothWinIn.setToolTip("""The size of the window for smoothing the results. 
Must be an odd number.
A value of 0 means no smoothing but can only be set to zero for displacement.""")
        self.gridLayout.addWidget(self.smoothWinIn, 1, 1, 1, 1)

        # The smoothing order input and label
        self.smoothOrderLab = QLabel(self)
        self.smoothOrderLab.setText("Smoothing Order:")
        self.gridLayout.addWidget(self.smoothOrderLab, 1, 2, 1, 1)

        self.smoothOrderIn = QLineEdit(self)
        self.smoothOrderIn.setText(str(self.resultsSmoothOrder))
        smoothOrderValidator = ClampingIntValidator()
        smoothOrderValidator.setBottom(0)
        self.smoothOrderIn.setValidator(smoothOrderValidator)
        self.smoothOrderIn.setToolTip(
            "Order of the Savitzky-Golay smoothing polynomial.")
        self.gridLayout.addWidget(self.smoothOrderIn, 1, 3, 1, 1)

        # The alpha input and label
        self.alphaLab = QLabel(self)
        self.alphaLab.setText("Alpha:")
        self.gridLayout.addWidget(self.alphaLab, 2, 0, 1, 1)

        self.alphaIn = QLineEdit(self)
        self.alphaIn.setText(str(self.resultsAlpha))
        alphaValidator = ClampingDblValidator(0.0, 1.0)
        self.alphaIn.setValidator(alphaValidator)
        self.alphaIn.setToolTip(
            "The transparency of the contour plot. Must be between 0 and 1.")
        self.gridLayout.addWidget(self.alphaIn, 2, 1, 1, 1)

        self.verticalLayout.addLayout(self.gridLayout)

        # Creating and adding the Submit Graph button
        self.submitGraphBut = QPushButton("Submit Graph", self)
        self.verticalLayout.addWidget(
            self.submitGraphBut, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)

        # Creating a spacer to neaten up the layout
        spacer = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum,
                             QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacer)

        # Connecting the input fields to the resultsConChanged method
        self.resultsSelector.currentIndexChanged.connect(
            self.resultsConChanged)

        self.submitGraphBut.clicked.connect(self.submitGraph)

    # ------------------------------------------------------------------------------
    # Function that is called to indicate that the data was changed by the user
    def resultsConChanged(self):

        # Update items in the components input field
        self.compIn.clear()
        self.compLab.setEnabled(True)
        self.compIn.setEnabled(True)
        self.smoothWinLab.setEnabled(True)
        self.smoothWinIn.setEnabled(True)
        self.smoothOrderLab.setEnabled(True)
        self.smoothOrderIn.setEnabled(True)

        if int(self.resultsSelector.currentIndex()) == self.DISP_INDEX:
            for index, e in enumerate(sdpp.DispComp):
                self.compIn.addItem(f"{e.display_name}")
        elif int(self.resultsSelector.currentIndex()) == self.STRAIN_INDEX:
            for index, e in enumerate(sdpp.StrainComp):
                self.compIn.addItem(f"{e.display_name}")
        elif int(self.resultsSelector.currentIndex()) == self.CORELLATION_INDEX:
            self.compLab.setEnabled(False)
            self.compIn.setEnabled(False)
            self.smoothWinLab.setEnabled(False)
            self.smoothWinIn.setEnabled(False)
            self.smoothOrderLab.setEnabled(False)
            self.smoothOrderIn.setEnabled(False)

    # ------------------------------------------------------------------------------
    # Function to submit the graph request
    def submitGraph(self):

        # Getting the required parameters
        alpha = float(self.alphaIn.text())
        smoothWindow = int(self.smoothWinIn.text())
        smoothOrder = int(self.smoothOrderIn.text())
        imgPair = self.parent.numImagePairs - self.imgPairIn.currentIndex() - 1

        # Plotting the graphs
        try:
            matplotlib.pyplot.close()

            # Plotting the displacement contour
            if self.resultsSelector.currentIndex() == self.DISP_INDEX:
                dispComp = getattr(sdpp.DispComp, sdpp.DispComp._member_names_[
                    self.compIn.currentIndex()])
                figure = sdpp.plotDispContour(self.parent.parent.savePath, imgPair=imgPair, dispComp=dispComp,
                                              alpha=alpha, plotImage=True,
                                              showPlot=False, fileName='',
                                              smoothWindow=smoothWindow, smoothOrder=smoothOrder,
                                              return_fig=True)
                showGraph(self, figure, self.verticalLayout)

            # Plotting the strain contour
            elif self.resultsSelector.currentIndex() == self.STRAIN_INDEX:
                strainComp = getattr(sdpp.StrainComp, sdpp.StrainComp._member_names_[
                    self.compIn.currentIndex()])
                figure = sdpp.plotStrainContour(self.parent.parent.savePath, imgPair=imgPair, strainComp=strainComp,
                                                alpha=alpha, plotImage=True,
                                                showPlot=False, fileName='',
                                                smoothWindow=smoothWindow, smoothOrder=smoothOrder,
                                                return_fig=True)
                showGraph(self, figure, self.verticalLayout)

            # Plotting the correlation contour
            elif self.resultsSelector.currentIndex() == self.CORELLATION_INDEX:
                figure = sdpp.plotZNCCContour(self.parent.parent.savePath, imgPair=imgPair,
                                              alpha=0.75, plotImage=True, showPlot=False,
                                              fileName='', return_fig=True)
                showGraph(self, figure, self.verticalLayout)

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred: {str(e)}")


class ResultsUILineCut(QWidget):
    """ Class for the results cutplot UI: Defines the layout and widgets for the 
        results cutplot tab
    """

    # Define all the variables to store the user input
    resultsImgPair = 0
    resultsSmoothWin = 3
    resultsSmoothOrder = 2
    resultsComp = 0
    resultsCutComp = 1
    resultsGridLines = True
    resultsInterp = False

    DISP_INDEX = 0
    STRAIN_INDEX = 1

    # ------------------------------------------------------------------------------
    # Function to initialize the UI
    def __init__(self, parent):

        super().__init__(parent)
        self.parent = parent

        # The main layout of the window
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setSpacing(10)

        # Add the results selector combo box
        self.resultsSelector = QComboBox(self)
        boxItems = ["Displacement", "Strain"]
        for item in boxItems:
            self.resultsSelector.addItem(item)
        self.verticalLayout.addWidget(
            self.resultsSelector, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        # Now the grid layout
        gridLayout = QGridLayout()
        gridLayout.setHorizontalSpacing(10)
        gridLayout.setVerticalSpacing(10)

        # The cut value label and input
        cutValLab = QLabel(self)
        cutValLab.setText("Cut Values:")
        gridLayout.addWidget(cutValLab, 1, 2, 1, 1)

        self.cutValIn = QLineEdit(self)
        self.cutValIn.setText(str(self.parent.parent.settings.ROI[1]))
        self.cutValIn.setToolTip("""The values at which to cut the data. 
Must be a comma separated list of integers.""")
        gridLayout.addWidget(self.cutValIn, 1, 3, 1, 1)

        # The image pair input and label
        imgPairLab = QLabel(self)
        imgPairLab.setText("Image Pair:")
        gridLayout.addWidget(imgPairLab, 0, 0, 1, 1)

        self.imgPairIn = QComboBox(self)
        numPairs = self.parent.numImagePairs
        for i in range(numPairs, 0, -1):
            self.imgPairIn.addItem(f"Image Pair {i}")
        gridLayout.addWidget(self.imgPairIn, 0, 1, 1, 1)

        # The display component input and label
        compLab = QLabel(self)
        compLab.setText("Display Component:")
        gridLayout.addWidget(compLab, 0, 2, 1, 1)

        self.compIn = QComboBox(self)
        self.compIn.setCurrentIndex(self.resultsComp)
        for index, e in enumerate(sdpp.DispComp):
            self.compIn.addItem(f"{e.display_name}")
        self.compIn.setToolTip("The component to plot.")
        gridLayout.addWidget(self.compIn, 0, 3, 1, 1)

        # The cut component input and label
        cutCompLab = QLabel(self)
        cutCompLab.setText("Cut Component:")
        gridLayout.addWidget(cutCompLab, 1, 0, 1, 1)

        self.cutCompIn = QComboBox(self)
        for index, e in enumerate(sdpp.CompID):
            if e.display_name != None:
                self.cutCompIn.addItem(f"{e.display_name}")
        self.cutCompIn.setCurrentIndex(self.resultsCutComp)
        self.cutCompIn.setToolTip(
            "The direction to cut the data along.")
        gridLayout.addWidget(self.cutCompIn, 1, 1, 1, 1)

        # The smoothing window size input and label
        smoothWinLab = QLabel(self)
        smoothWinLab.setText("Smoothing Window:")
        gridLayout.addWidget(smoothWinLab, 2, 0, 1, 1)

        self.smoothWinIn = QLineEdit(self)
        self.smoothWinIn.setText(str(self.resultsSmoothWin))
        smoothWindowValidator = OddNumberValidator(0, None)
        self.smoothWinIn.setValidator(smoothWindowValidator)
        self.smoothWinIn.setToolTip("""The size of the window for smoothing the results. 
Must be an odd number.
A value of 0 means no smoothing but can only be set to zero for displacement.""")
        gridLayout.addWidget(self.smoothWinIn, 2, 1, 1, 1)

        # The smoothing order input and label
        smoothOrderLab = QLabel(self)
        smoothOrderLab.setText("Smoothing Order:")
        gridLayout.addWidget(smoothOrderLab, 2, 2, 1, 1)

        self.smoothOrderIn = QLineEdit(self)
        self.smoothOrderIn.setText(str(self.resultsSmoothOrder))
        smoothOrderValidator = ClampingIntValidator()
        smoothOrderValidator.setBottom(1)
        self.smoothOrderIn.setValidator(smoothOrderValidator)
        self.smoothOrderIn.setToolTip(
            "Order of the Savitzky-Golay smoothing polynomial.")
        gridLayout.addWidget(self.smoothOrderIn, 2, 3, 1, 1)

        # The interpolation checkbox and label
        interpLab = QLabel(self)
        interpLab.setText("Interpolate:")
        gridLayout.addWidget(interpLab, 4, 0, 1, 1)

        self.interpIn = QCheckBox(self)
        self.interpIn.setChecked(self.resultsInterp)
        self.interpIn.setToolTip(
            "Whether to interpolate the data.")
        gridLayout.addWidget(self.interpIn, 4, 1, 1, 1)

        # The grid lines checkbox and label
        gridLinesLab = QLabel(self)
        gridLinesLab.setText("Grid Lines:")
        gridLayout.addWidget(gridLinesLab, 4, 2, 1, 1)

        self.gridLinesIn = QCheckBox(self)
        self.gridLinesIn.setChecked(self.resultsGridLines)
        self.gridLinesIn.setToolTip(
            "Whether to show grid lines on the plot.")
        gridLayout.addWidget(self.gridLinesIn, 4, 3, 1, 1)

        self.verticalLayout.addLayout(gridLayout)

        # Creating and adding the Submit Graph button
        self.submitGraphBut = QPushButton("Submit Graph", self)
        self.verticalLayout.addWidget(
            self.submitGraphBut, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)

        # Creating a spacer to neaten up the layout
        spacer = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum,
                             QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacer)

        # Connecting the Submit Graph button to the submitGraph method
        self.resultsSelector.currentIndexChanged.connect(
            self.resultsCutChanged)
        self.submitGraphBut.clicked.connect(self.submitGraph)

    # ------------------------------------------------------------------------------
    # Function that is called to indicate that the data was changed by the user
    def resultsCutChanged(self):
        # Update items in the components input field
        self.compIn.clear()
        if int(self.resultsSelector.currentIndex()) == self.DISP_INDEX:
            for index, e in enumerate(sdpp.DispComp):
                self.compIn.addItem(f"{e.display_name}")
        elif int(self.resultsSelector.currentIndex()) == self.STRAIN_INDEX:
            for index, e in enumerate(sdpp.StrainComp):
                self.compIn.addItem(f"{e.display_name}")

    # ------------------------------------------------------------------------------
    # Function that is called to generate the graph based on user input
    def submitGraph(self):
        # Getting the required parameters
        imgPair = self.parent.numImagePairs - self.imgPairIn.currentIndex() - 1
        cutComp = getattr(sdpp.CompID, sdpp.CompID._member_names_[
                          self.cutCompIn.currentIndex()])
        cutValues = [int(i)
                     for i in self.cutValIn.text().split(",")]
        gridlines = self.gridLinesIn.isChecked()
        smoothWindow = int(self.smoothWinIn.text())
        smoothOrder = int(self.smoothOrderIn.text())
        interpolate = self.interpIn.isChecked()

        try:
            matplotlib.pyplot.close()

            # Plotting the displacement cut line
            if self.resultsSelector.currentIndex() == self.DISP_INDEX:
                dispComp = getattr(sdpp.DispComp, sdpp.DispComp._member_names_[
                    self.compIn.currentIndex()])
                figure = sdpp.plotDispCutLine(self.parent.parent.savePath, imgPair=imgPair, dispComp=dispComp,
                                              cutComp=cutComp, cutValues=cutValues, gridLines=gridlines,
                                              showPlot=False, fileName='', smoothWindow=smoothWindow, smoothOrder=smoothOrder,
                                              interpolate=interpolate, return_fig=True)

                showGraph(self, figure, self.verticalLayout)

            # Plotting the strain cut line
            elif self.resultsSelector.currentIndex() == self.STRAIN_INDEX:
                strainComp = getattr(sdpp.StrainComp, sdpp.StrainComp._member_names_[
                    self.compIn.currentIndex()])
                figure = sdpp.plotStrainCutLine(self.parent.parent.savePath, imgPair=imgPair, strainComp=strainComp, cutComp=cutComp,
                                                cutValues=cutValues, gridLines=gridlines, showPlot=False,
                                                fileName='', smoothWindow=smoothWindow, smoothOrder=smoothOrder,
                                                interpolate=interpolate, return_fig=True)

                showGraph(self, figure, self.verticalLayout)
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred: {str(e)}")

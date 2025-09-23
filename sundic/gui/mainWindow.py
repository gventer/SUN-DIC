import os
import sys
import time
from enum import IntEnum

from PyQt6.QtCore import Qt, QTimer, QSize, QRect
from PyQt6.QtGui import QPainter, QAction, QPixmap, QIcon, QFont, QFontDatabase
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QFileDialog, QMessageBox, QStyle, QStackedLayout,
    QSplashScreen, QPushButton, QMenuBar, QMenu, QStatusBar, QWidget, QFrame,
    QGridLayout, QVBoxLayout, QSpacerItem, QSizePolicy, QTabWidget
)

from sundic.gui.settingsWidget import SettingsUI
from sundic.gui.imageSetWidget import ImageSetUI
from sundic.gui.roiDefWidget import ROIDefUI
from sundic.gui.analysisWidget import AnalysisUI
from sundic.gui.resultsWidget import ResultsUI
from sundic.gui.aboutWidget import AboutDialog
from sundic.gui.buttonWidget import LeftIconButton

import sundic.settings as sdset
import sundic.util.datafile as dataFile
import sundic as sd


class IntConst(IntEnum):
    """ Integer constants for the different tabs in the main window
    """
    SETTINGS_TAB = 0
    IMAGESET_TAB = 1
    ROI_TAB = 2
    ANALYSIS_TAB = 3
    RESULTS_TAB = 4


class UIMainWindow(object):
    """ Main Window UI Class
    """

    # ------------------------------------------------------------------------------
    # Setup the main UI
    def setupMainUI(self, parent):

        self.parent = parent

        # Set a constant for the icon size
        ICON_SIZE = 32

        # Get the path to the icons
        scriptDir = os.path.dirname(os.path.abspath(__file__))
        iconPath = os.path.join(scriptDir, "icons")

        # Define custom bottom style for buttons
        buttonStyle = "QPushButton{\n" +\
            "border: 2px  solid  rgb(0, 0, 0);\n" +\
            "background: rgb(255, 255, 255);\n" +\
            "border-style: outset;\n" +\
            "border-width: 1px 1px 1px 1px;\n" +\
            "border-radius: 0px;\n" +\
            "color: black;\n" +\
            "text-align:left;\n" +\
            "padding-left: 20px;\n" +\
            "padding: 5px 5px 5px 5px;   \n" +\
            "}\n" +\
            "QPushButton:checked {\n" +\
            "color: gray;\n" +\
            "border-style: inset;\n" +\
            "background: qradialgradient(\n" +\
            "cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,\n" +\
            "radius: 1.35, stop: 0 #fff, stop: 1 #90D5FF\n" +\
            ");\n" +\
            "}\n" +\
            "QPushButton:disabled {\n" +\
            "border-style: outset;\n" +\
            "background: qradialgradient(\n" +\
            "cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,\n" +\
            "radius: 1.35, stop: 0 #fff, stop: 1 #ddd\n" +\
            ");\n" +\
            "color: gray;\n" +\
            "}\n"

        # Size the main Window and set the title
        self.parent.resize(1000, 800)
        self.parent.setWindowTitle("SUN DIC")

        # Setup the font for the application
        fontPath = os.path.join(
            scriptDir, "Fonts", "Figtree", "Figtree-VariableFont_wght.ttf")
        QFontDatabase.addApplicationFont(fontPath)
        font = QFont()
        font.setFamily("Figtree Light")
        fontSize = font.pointSize()
        self.parent.setFont(font)
        QApplication.setFont(font)

        # Create a larger font for the buttons
        buttonFont = QFont()
        buttonFont.setFamily("Figtree Light")
        buttonFont.setPointSize(int(1.33*fontSize))

        # Set window properties
        self.parent.setWindowOpacity(1.0)
        self.parent.setAutoFillBackground(False)
        self.parent.setStyleSheet("QMainWindow{\n"
                                  "background-color: rgb(255, 255, 255);\n"
                                  "}")
        self.parent.setTabShape(QTabWidget.TabShape.Rounded)

        # Central widget
        centralwidget = QWidget(self.parent)
        centralwidget.setEnabled(True)

        # The main layout
        gridLayout_2 = QGridLayout(centralwidget)
        gridLayout = QGridLayout()
        gridLayout.setContentsMargins(0, 0, 0, 0)
        gridLayout.setSpacing(0)
        verticalLayout = QVBoxLayout()
        verticalLayout.setSpacing(0)

        # Define the settings button
        self.settingsBut = LeftIconButton(
            centralwidget, stylesheet=buttonStyle)
        self.settingsBut.setText("Settings")
        self.settingsBut.setEnabled(True)
        self.settingsBut.setFont(buttonFont)
        self.settingsBut.setDefault(True)
        self.settingsBut.setStyleSheet(buttonStyle)
        self.settingsBut.setCheckable(True)
        self.settingsBut.setAutoExclusive(True)
        self.settingsBut.clicked.connect(self.settingsAction)
        self.settingsBut.setIcon(
            QIcon(os.path.join(iconPath, "settings.png")))
        self.settingsBut.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        verticalLayout.addWidget(self.settingsBut)

        # Define the image set button
        self.imageSetBut = LeftIconButton(
            centralwidget, stylesheet=buttonStyle)
        self.imageSetBut.setText("Image Set")
        self.imageSetBut.setFont(buttonFont)
        self.imageSetBut.setStyleSheet(buttonStyle)
        self.imageSetBut.setCheckable(True)
        self.imageSetBut.setAutoExclusive(True)
        self.imageSetBut.setIcon(
            QIcon(os.path.join(iconPath, "imageSet.png")))
        self.imageSetBut.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.imageSetBut.clicked.connect(self.imageSetAction)
        verticalLayout.addWidget(self.imageSetBut)

        # Define the ROI button
        self.roiBut = LeftIconButton(
            centralwidget, stylesheet=buttonStyle)
        self.roiBut.setText("ROI Definition")
        self.roiBut.setFont(buttonFont)
        self.roiBut.setStyleSheet(buttonStyle)
        self.roiBut.setCheckable(True)
        self.roiBut.setAutoExclusive(True)
        self.roiBut.setIcon(
            QIcon(os.path.join(iconPath, "roi.png")))
        self.roiBut.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.roiBut.clicked.connect(self.roiDefAction)
        verticalLayout.addWidget(self.roiBut)

        # Define the analysis button
        self.analysisBut = LeftIconButton(
            centralwidget, stylesheet=buttonStyle)
        self.analysisBut.setText("Analysis")
        self.analysisBut.setFont(buttonFont)
        self.analysisBut.setStyleSheet(buttonStyle)
        self.analysisBut.setCheckable(True)
        self.analysisBut.setAutoExclusive(True)
        self.analysisBut.setIcon(
            QIcon(os.path.join(iconPath, "analysis.png")))
        self.analysisBut.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.analysisBut.clicked.connect(self.analysisAction)
        verticalLayout.addWidget(self.analysisBut)

        # Define the results button
        self.resultsBut = LeftIconButton(
            centralwidget, stylesheet=buttonStyle)
        self.resultsBut.setText("Results")
        self.resultsBut.setFont(buttonFont)
        self.resultsBut.setStyleSheet(buttonStyle)
        self.resultsBut.setCheckable(True)
        self.resultsBut.setAutoExclusive(True)
        self.resultsBut.setIcon(
            QIcon(os.path.join(iconPath, "results.png")))
        self.resultsBut.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.resultsBut.clicked.connect(self.resultsAction)
        verticalLayout.addWidget(self.resultsBut)

        # Some spacers
        spacerItem = QSpacerItem(20, 40,
                                 QSizePolicy.Policy.Minimum,
                                 QSizePolicy.Policy.Expanding)
        verticalLayout.addItem(spacerItem)

        gridLayout.addLayout(verticalLayout, 0, 0, 1, 1)

        # The main frame where the other widgets will be placed
        self.mainFrame = QFrame(centralwidget)
        self.mainFrame.setEnabled(True)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.mainFrame.sizePolicy().hasHeightForWidth())
        self.mainFrame.setSizePolicy(sizePolicy)
        self.mainFrame.setMinimumSize(QSize(800, 4))
        self.mainFrame.setAutoFillBackground(False)
        self.mainFrame.setFrameShape(QFrame.Shape.Panel)
        self.mainFrame.setFrameShadow(QFrame.Shadow.Plain)
        self.mainFrame.setLineWidth(1)

        gridLayout.addWidget(self.mainFrame, 0, 1, 1, 1)
        gridLayout_2.addLayout(gridLayout, 0, 0, 1, 1)

        # Add the central widget to the main window
        self.parent.setCentralWidget(centralwidget)

        # Setup the menu bar
        menubar = QMenuBar(self.parent)
        self.parent.setMenuBar(menubar)

        # Setup the menus
        menuFile = QMenu(menubar)
        menuFile.setTitle("File")
        menuAbout = QMenu(menubar)
        menuAbout.setTitle("About")

        # Setup the status bar
        statusbar = QStatusBar(self.parent)
        self.parent.setStatusBar(statusbar)

        # Create the actions
        self.actionOpen = QAction(self.parent)
        self.actionOpen.setText("Open")
        openIcon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton)
        self.actionOpen.setIcon(openIcon)
        self.actionOpen.triggered.connect(self.openAction)
        self.actionSave = QAction(self.parent)
        self.actionSave.setText("Save")
        saveIcon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        self.actionSave.setIcon(saveIcon)
        self.actionSave.triggered.connect(self.saveAction)

        self.actionSave_as = QAction(self.parent)
        self.actionSave_as.setText("Save as")
        self.actionSave_as.triggered.connect(self.saveAsAction)

        self.actionNew = QAction(self.parent)
        self.actionNew.setText("New")
        newIcon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
        self.actionNew.setIcon(newIcon)
        self.actionNew.triggered.connect(self.newAction)

        self.actionExit = QAction(self.parent)
        self.actionExit.setText("Exit")
        self.actionExit.triggered.connect(self.exitAction)

        self.actionAbout = QAction(self.parent)
        self.actionAbout.setText("About")
        aboutIcon = self.style().standardIcon(
            QStyle.StandardPixmap.SP_MessageBoxInformation)
        self.actionAbout.setIcon(aboutIcon)
        self.actionAbout.triggered.connect(self.aboutAction)

        # Add the actions to the menus
        menuFile.addAction(self.actionNew)
        menuFile.addAction(self.actionOpen)
        menuFile.addAction(self.actionSave)
        menuFile.addAction(self.actionSave_as)
        menuFile.addAction(self.actionExit)

        menuAbout.addAction(self.actionAbout)

        # Add the menus to the menubar
        menubar.addAction(menuFile.menuAction())
        menubar.addAction(menuAbout.menuAction())

    # ------------------------------------------------------------------------------
    # Show the Settings tab

    def settingsAction(self):
        self.parent.settingsUI.setData(self.parent.settings)
        self.stackedLayout.setCurrentIndex(IntConst.SETTINGS_TAB)

    # ------------------------------------------------------------------------------
    # Show the Image Set tab
    def imageSetAction(self):
        self.parent.imageSetUI.setData(self.parent.settings)
        self.stackedLayout.setCurrentIndex(IntConst.IMAGESET_TAB)

    # ------------------------------------------------------------------------------
    # Show the ROI Definition tab
    def roiDefAction(self):
        self.parent.roiDefUI.setData(self.parent.settings)
        self.stackedLayout.setCurrentIndex(IntConst.ROI_TAB)

    # ------------------------------------------------------------------------------
    # Show the Analysis tab
    def analysisAction(self):
        self.parent.analysisUI.setData(self.parent.settings)
        self.stackedLayout.setCurrentIndex(IntConst.ANALYSIS_TAB)

    # ------------------------------------------------------------------------------
    # Show the Results tab
    def resultsAction(self):
        self.stackedLayout.setCurrentIndex(IntConst.RESULTS_TAB)

    # ------------------------------------------------------------------------------
    # Open a new project in the GUI
    def openAction(self):

        # Check if we have unsaved changes and warn the user
        if self.parent.savedFlag == False:
            warn = QMessageBox.question(
                self.parent, "Warning", "You have unsaved changes. Do you want to proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if warn == QMessageBox.StandardButton.No:
                return

        # Ask the user to select a file for opening
        options = QFileDialog.Option.ReadOnly
        filePath, _ = QFileDialog.getOpenFileName(
            self.parent, "Open File", "", "SUN-DIC Binary Files (*.sdic);;All Files (*)",
            options=options)

        if filePath:
            # Setup the parent class variables
            self.parent.savedFlag = True
            self.parent.savePath = filePath
            self.parent.updateWindowTitle()
            self.parent.settings = sdset.Settings.fromMsgPackFile(filePath)
            self.parent.settings.ImageFolder = self.parent.getAbsImageFolder(filePath,
                                                                             self.parent.settings.ImageFolder)

            # See if we should enable the results button - only if results exist
            resultsFile = dataFile.DataFile.openReader(filePath)
            if resultsFile.containsResults():
                self.parent.resultsBut.setEnabled(True)
            else:
                self.parent.resultsBut.setEnabled(False)

            # Set the data on the UI tabs
            self.parent.settingsBut.setChecked(True)
            self.parent.settingsAction()

        # Create a new results UI
        self.newResultsUIWidget()

    # ------------------------------------------------------------------------------
    # Save the current settings to the SUN-DIC file
    def saveAction(self):

        # Check if results already exist for this file.  If they do, warn the user
        # that this save action will delete the existing results
        if self.parent.savePath is not None and os.path.isfile(self.parent.savePath):
            resultsFile = dataFile.DataFile.openReader(self.parent.savePath)
            if resultsFile.containsResults():
                warn = QMessageBox.question(
                    self.parent, "Warning", "Results already exist for this file. If you choose to save the new settings, the previous results will be deleted. Do you want to proceed?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                if warn == QMessageBox.StandardButton.No:
                    return

        # If we don't have a save path yet, call the saveAs action
        if self.parent.savePath is None:
            # self.parent.savedFlag = True  # do this to avoid the saveAsAction warning
            self.saveAsAction()

        # Write the settings and results to the new file
        else:
            f = dataFile.DataFile.openWriter(self.parent.savePath)
            f.writeHeading(self.parent.settings)
            f.close()
            self.parent.savedFlag = True
            self.parent.updateWindowTitle()

        # Create a new results UI
        self.newResultsUIWidget()

    # ------------------------------------------------------------------------------
    # Save the current settings to a new SUN-DIC file
    def saveAsAction(self):

        # Now ask the user to select a file for saving
        fileName, _ = QFileDialog.getSaveFileName(self.parent,
                                                  "Save File", "", "SUNDIC Binary File(*.sdic)")

        # Check if we got a filename
        if fileName:
            if not fileName.endswith('.sdic'):
                fileName = fileName+'.sdic'

            # Write the settings to the new file
            newFile = dataFile.DataFile.openWriter(fileName)
            newFile.writeHeading(self.parent.settings)

            # Copy any old data over if there are any
            if self.parent.savePath is not None and os.path.isfile(self.parent.savePath):
                oldDataFile = dataFile.DataFile.openReader(
                    self.parent.savePath)
                if oldDataFile.containsResults():
                    numPairs = oldDataFile.getNumImagePairs()
                    for pair in range(numPairs):
                        oldData = oldDataFile.readSubSetData(pair)
                        newFile.writeSubSetData(pair, oldData)
                    oldDataFile.close()

            # Close the file
            newFile.close()

            # Set the parent class variables
            self.parent.savePath = fileName
            self.parent.savedFlag = True
            self.parent.updateWindowTitle()

        # Create a new results UI
        self.newResultsUIWidget()

    # ------------------------------------------------------------------------------
    # Create a new project in the GUI
    def newAction(self):

        # Check if we have unsaved changes and warn the user
        if self.parent.savedFlag == False:
            warn = QMessageBox.question(
                self.parent, "Warning", "You have unsaved changes. Do you want to proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if warn == QMessageBox.StandardButton.No:
                return

        # Create a new settings object and set the data on the UI tabs
        self.parent.settings = sdset.Settings()
        self.settingsBut.setChecked(True)
        self.settingsAction()
        self.parent.savedFlag = True
        self.parent.savePath = None
        self.parent.updateWindowTitle()
        self.parent.resultsBut.setEnabled(False)

        # Create a new results UI
        self.newResultsUIWidget()

    # ------------------------------------------------------------------------------
    # Method to properly destroy and recreate the resultsUI widget
    def newResultsUIWidget(self):

        # Check if we already have a resultsUI and delete it if we do
        if self.parent.resultsUI is not None:
            self.parent.stackedLayout.removeWidget(self.parent.resultsUI)
            self.parent.resultsUI.setParent(None)
            self.parent.resultsUI.deleteLater()
            self.parent.resultsUI = None

        # Create a new resultsUI and add it to the stacked layout
        self.parent.resultsUI = ResultsUI(self.parent)
        self.parent.stackedLayout.addWidget(self.parent.resultsUI)

        # Check if we should enable the results button - only if results exist
        if self.parent.savePath is not None and os.path.isfile(self.parent.savePath):
            resultsFile = dataFile.DataFile.openReader(self.parent.savePath)
            if resultsFile.containsResults():
                self.parent.resultsBut.setEnabled(True)
            else:
                self.parent.resultsBut.setEnabled(False)
        else:
            self.parent.resultsBut.setEnabled(False)

    # ------------------------------------------------------------------------------
    # Create an exit action
    def exitAction(self):
        self.parent.close()

    # ------------------------------------------------------------------------------
    # Display the about dialog
    def aboutAction(self):
        version = sd.version.__version__
        dlg = AboutDialog(self.parent, version=version)
        dlg.exec()


class mainProgram(QMainWindow, UIMainWindow):
    """ The main program class that contains the different tabs
    """

    # The different UI tabs
    settingsUI = None
    imageSetUI = None
    roiDefUI = None
    analysisUI = None
    resultsUI = None

    # The saved status and path to the current SUN-DIC file
    savedFlag = True
    savePath = None

    # The settings object that is passed around the different tabs
    settings = None

    # ------------------------------------------------------------------------------
    # The class constructor
    def __init__(self, parent=None):

        # Inherit from the aforementioned class and set up the gui
        super().__init__()

        self.setupMainUI(self)

        # Process events to make sure the splash screen is shown
        QApplication.processEvents()

        # Now show the main window
        self.show()

        # The default settings object
        self.settings = sdset.Settings()

        # Setup the tabs to show
        self.settingsUI = SettingsUI(self)
        self.settingsBut.setChecked(True)
        self.settingsUI.setData(self.settings)

        self.imageSetUI = ImageSetUI(self)
        self.roiDefUI = ROIDefUI(self)
        self.analysisUI = AnalysisUI(self)
        self.resultsUI = ResultsUI(self)

        # Setup local class variables
        self.savePath = None
        self.savedFlag = True
        self.updateWindowTitle()

        # Disable the results button until results are available
        self.resultsBut.setEnabled(False)

        # Create a stacked layout for the items to show
        self.stackedLayout = QStackedLayout(self.mainFrame)
        self.stackedLayout.addWidget(self.settingsUI)
        self.stackedLayout.addWidget(self.imageSetUI)
        self.stackedLayout.addWidget(self.roiDefUI)
        self.stackedLayout.addWidget(self.analysisUI)
        self.stackedLayout.addWidget(self.resultsUI)

        self.settingsBut.setChecked(True)

    # ------------------------------------------------------------------------------
    # Define the close event - what to do when we close the application
    def closeEvent(self, event):

        # Check if we have unsaved changes and warn the user
        try:
            if self.savedFlag == False:
                reply = QMessageBox.question(self, 'Window Close',
                                             'Are you sure you want to exit without saving?',
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Save, QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    event.ignore()
                    return
                elif reply == QMessageBox.StandardButton.Save:
                    self.saveAction()
                    event.accept()
        except Exception as e:
            print(f"Exception during closeEvent: {e}")
            event.accept()

        # Stop and wait for the analysis thread if it's running
        if hasattr(self, 'analysisUI') and hasattr(self.analysisUI, 'worker'):
            worker = self.analysisUI.worker
            if worker is not None and worker.isRunning():
                worker.stop()
                worker.wait()

        super().closeEvent(event)

    # ------------------------------------------------------------------------------
    # Update the window title based on the save status
    def updateWindowTitle(self):

        # No file attached yet
        if self.savePath is None:
            if self.savedFlag:
                self.setWindowTitle("SUN-DIC")
            else:
                self.setWindowTitle("* SUN-DIC")

        # File already attached so display the name
        else:
            if self.savedFlag:
                self.setWindowTitle(
                    str(os.path.basename(self.savePath)) + " - SUN-DIC")
            else:
                self.setWindowTitle(
                    str(os.path.basename(self.savePath)) + "* - SUN-DIC")

    # ------------------------------------------------------------------------------
    # Get the absolute path to the image folder based on the results file location
    def getAbsImageFolder(self, resultsFile, imageFolder):

        # Check if the imageFolder is an absolute path
        if os.path.isabs(imageFolder):
            return imageFolder

        # If not, make it an absolute path that is relative to the resultsFile
        elif imageFolder is not None and os.path.isabs(resultsFile):
            return os.path.abspath(os.path.join(os.path.dirname(resultsFile), imageFolder))

        # Otherwise, return the image folder as is
        else:
            return imageFolder


# ------------------------------------------------------------------------------
# The main function that gets called when the program is executed
def main():
    # Make an object of the class and execute it
    QApplication.setStyle("fusion")
    app = QApplication(sys.argv)

    # Load splash image
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    iconsPath = os.path.join(scriptDir, "icons", "splashScreen.png")
    splash = QSplashScreen(QPixmap(iconsPath),
                           Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    # Increase the font size for the splash screen
    font = splash.font()
    font.setPointSize(16)
    splash.setFont(font)
    splash.showMessage(
        "Version "+sd.version.__version__,
        alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
        color=Qt.GlobalColor.black
    )

    # Create and show the main window
    window = mainProgram()
    window.show()
    time.sleep(1.5)
    splash.finish(window)

    # Exit the window cleanly
    return app.exec()


# ------------------------------------------------------------------------------
# Call the main function if this file is executed
if __name__ == "__main__":
    sys.exit(main())

import socket  # noqa: N999
import sys
import time

from PyQt6 import QtGui
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import sundic.sundic as sd
from sundic.gui.validators import ClampingIntValidator


class AnalysisUI(QWidget):
    """Class for the analysis UI: Defines the layout and widgets for the analysis tab
    and contains functions to get and set the data in the settings object.
    """

    # ------------------------------------------------------------------------------
    # Initialize the analysis UI
    def __init__(self, parent):

        super().__init__(parent)
        self.parent = parent

        # Set the class variables
        verticalLayout = QVBoxLayout(self)
        verticalLayout.setContentsMargins(20, 20, 20, 20)
        verticalLayout.setSpacing(0)

        gridLayout = QGridLayout()
        gridLayout.setContentsMargins(-1, -1, -1, 5)
        gridLayout.setHorizontalSpacing(10)
        gridLayout.setVerticalSpacing(10)

        # Number of CPUs input and label
        cpuLab = QLabel(self)
        cpuLab.setText("CPU Count:")
        gridLayout.addWidget(cpuLab, 1, 0, 1, 1)

        self.cpuIn = QLineEdit(self)
        cpuValidator = ClampingIntValidator()
        cpuValidator.setBottom(1)
        self.cpuIn.setValidator(cpuValidator)
        self.cpuIn.setText("1")
        self.cpuIn.setToolTip("""The number of CPU cores to use for the analysis. 
Must be larger than or equal to 1.""")
        gridLayout.addWidget(self.cpuIn, 1, 1, 1, 1)

        # The debug level input and label
        debugLab = QLabel(self)
        debugLab.setText("Debug Level:")
        gridLayout.addWidget(debugLab, 0, 0, 1, 1)

        self.debugIn = QComboBox(self)
        debugInItems = [
            "0 - No Debugging",
            "1 - Debugging",
            "2 - Debugging with extra information",
        ]
        for item in debugInItems:
            self.debugIn.addItem(item)
        gridLayout.addWidget(self.debugIn, 0, 1, 1, 1)

        # Compress data file checkbox and label
        self.compressCheck = QCheckBox(self)
        self.compressCheck.setText("Compress Data File")
        self.compressCheck.setToolTip("""If checked, the results data file will be compressed 
to reduce its size.""")
        gridLayout.addWidget(self.compressCheck, 0, 2, 1, 2)

        # Data ouput dropdown and label
        self.dataOutLab = QLabel(self)
        self.dataOutLab.setText("Data Output:")
        gridLayout.addWidget(self.dataOutLab, 1, 2, 1, 1)

        self.dataOutIn = QComboBox(self)
        self.dataOutIn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.dataOutIn.setToolTip("""Select the type of data to output to the results file.
DispOnly is all that is required for post-processing.  All will output all internal data, 
which is only required by advanced users and for specialized debugging.""")
        dataOutInItems = ["DispOnly", "All"]
        for item in dataOutInItems:
            self.dataOutIn.addItem(item)
        gridLayout.addWidget(self.dataOutIn, 1, 3, 1, 1)

        spacerItem1 = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        gridLayout.addItem(spacerItem1, 0, 2, 1, 1)
        spacerItem2 = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        gridLayout.addItem(spacerItem2, 1, 2, 1, 1)

        verticalLayout.addLayout(gridLayout)
        spacerItem3 = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        verticalLayout.addItem(spacerItem3)

        horizontalLayout_2 = QHBoxLayout()

        # Program output label and text box
        self.statusLab1 = QLabel(self)
        self.statusLab1.setText("Status: ")
        horizontalLayout_2.addWidget(self.statusLab1)
        self.statusLab2 = QLabel(self)
        self.statusLab2.setText(" ")
        horizontalLayout_2.addWidget(self.statusLab2)

        font = QtGui.QFont()
        font.setFamily("Courier New")
        self.progOut = QTextEdit(self)
        self.progOut.setFrameShape(QFrame.Shape.Panel)
        self.progOut.setFrameShadow(QFrame.Shadow.Plain)
        self.progOut.setFont(font)
        self.progOut.setReadOnly(True)
        self.progOut.setText("Program Output will be captured here..........")
        verticalLayout.addWidget(self.progOut)

        spacerItem4 = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        horizontalLayout_2.addItem(spacerItem4)

        spacerItem5 = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        verticalLayout.addItem(spacerItem5)
        verticalLayout.addLayout(horizontalLayout_2)
        spacerItem6 = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        verticalLayout.addItem(spacerItem6)

        horizontalLayout = QHBoxLayout()

        # The start and stop buttons
        self.startBut = QPushButton(self)
        self.startBut.setText("Start")
        horizontalLayout.addWidget(self.startBut)

        spacerItemH = QSpacerItem(
            10, 10, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum
        )
        horizontalLayout.addItem(spacerItemH)

        self.stopBut = QPushButton(self)
        self.stopBut.setText("Stop")
        self.stopBut.setEnabled(False)
        horizontalLayout.addWidget(self.stopBut)

        spacerItem7 = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        horizontalLayout.addItem(spacerItem7)
        verticalLayout.addLayout(horizontalLayout)

        # Connect the buttons
        self.startBut.clicked.connect(self.submitDIC)
        self.stopBut.clicked.connect(self.stopDIC)

        # Connecting to input
        self.debugIn.currentIndexChanged.connect(self.changedAnalysis)
        self.cpuIn.editingFinished.connect(self.changedAnalysis)
        self.compressCheck.stateChanged.connect(self.changedAnalysis)
        self.dataOutIn.currentIndexChanged.connect(self.changedAnalysis)

    # ------------------------------------------------------------------------------
    # Function to get the data from the settings UI and set it in the settings object
    def getData(self, settings):
        settings.DebugLevel = self.debugIn.currentIndex()
        settings.CPUCount = int(self.cpuIn.text())
        settings.DataCompression = self.compressCheck.isChecked()
        settings.DataSaveMode = self.dataOutIn.currentText()

    # ------------------------------------------------------------------------------
    # Function to get the data from this class and store it in the settings object
    def setData(self, settings):
        self.debugIn.blockSignals(True)
        self.compressCheck.blockSignals(True)
        self.dataOutIn.blockSignals(True)

        self.debugIn.setCurrentIndex(settings.DebugLevel)
        self.cpuIn.setText(str(settings.CPUCount))
        self.compressCheck.setChecked(settings.DataCompression)
        self.dataOutIn.setCurrentIndex(self.dataOutIn.findText(settings.DataSaveMode))

        self.debugIn.blockSignals(False)
        self.compressCheck.blockSignals(False)
        self.dataOutIn.blockSignals(False)

    # ------------------------------------------------------------------------------
    # Function that is called to indicate that the data was changed by the user
    def changedAnalysis(self):  # Saving User Input
        self.parent.savedFlag = False
        self.parent.updateWindowTitle()
        self.getData(self.parent.settings)

    # ------------------------------------------------------------------------------
    # Function to start the DIC analysis
    def submitDIC(self):

        # Save the current settings to file and run the analysis
        self.parent.savedFlag = False
        self.parent.updateWindowTitle()
        self.parent.saveAction()
        if self.parent.savedFlag is False:
            return

        # Print the settings to the output window
        self.progOut.setPlainText(self.parent.settings.__repr__())

        # Start the run
        # use tic/toc to time the analysis and print the time to the output window

        global START_TIME
        START_TIME = time.time()
        self.worker = PlanarDICWorker(
            settings=self.parent.parent.settings,
            resultsFile=self.parent.parent.savePath,
        )

        self.worker.progress.connect(self.appendProgress)
        self.worker.started.connect(self.startedRunOutput)
        self.worker.finished.connect(self.finishedRunOutput)
        self.worker.start()

    # ------------------------------------------------------------------------------
    # Method to append program output to the text box
    def appendProgress(self, text):
        text = text.rstrip("\n")
        if text:
            self.progOut.append(text)

    # ------------------------------------------------------------------------------
    # Method to handle the started signal
    def startedRunOutput(self, text):
        self.parent.settingsBut.setEnabled(False)
        self.parent.imageSetBut.setEnabled(False)
        self.parent.roiBut.setEnabled(False)
        self.parent.analysisBut.setEnabled(True)
        self.parent.resultsBut.setEnabled(False)

        self.startBut.setEnabled(False)
        self.stopBut.setEnabled(True)

        self.statusLab2.setText(text)
        self.statusLab2.setStyleSheet("color: blue")

    # ------------------------------------------------------------------------------
    # Method to handle the finished signal
    def finishedRunOutput(self, text):

        stop_time = time.time()
        elapsed_time = stop_time - START_TIME
        self.appendProgress(
            f"\nTime to complete the analysis: {elapsed_time:.2f} seconds"
        )

        self.parent.settingsBut.setEnabled(True)
        self.parent.imageSetBut.setEnabled(True)
        self.parent.roiBut.setEnabled(True)
        self.parent.analysisBut.setEnabled(True)
        self.parent.resultsBut.setEnabled(True)

        self.startBut.setEnabled(True)
        self.stopBut.setEnabled(False)

        self.statusLab2.setText(text)
        self.statusLab2.setStyleSheet("color: green")

        # Create a new and updated resultsUI
        self.parent.newResultsUIWidget()

    # ------------------------------------------------------------------------------
    # Function to stop the DIC analysis
    def stopDIC(self):

        # Stop the analysis
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()  # Wait for the thread to finish
            self.appendProgress("Analysis Stopped by User")

        # Reset the gui
        self.parent.settingsBut.setEnabled(True)
        self.parent.imageSetBut.setEnabled(True)
        self.parent.roiBut.setEnabled(True)
        self.parent.analysisBut.setEnabled(True)
        self.parent.resultsBut.setEnabled(False)

        self.startBut.setEnabled(True)
        self.stopBut.setEnabled(False)

        self.statusLab2.setText("Analysis Stopped")
        self.statusLab2.setStyleSheet("color: red")

        # Create a new and updated resultsUI
        self.parent.newResultsUIWidget()


class PlanarDICWorker(QThread):
    """Worker thread class for running the planar DIC analysis in the background
    without blocking the GUI.
    """

    # Define signals
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    started = pyqtSignal(str)

    # Is there an external ray server running?
    externalRay = False

    # ------------------------------------------------------------------------------
    # Initialize the worker thread
    def __init__(self, settings, resultsFile):
        super().__init__()
        self.settings = settings
        self.resultsFile = resultsFile
        self._isRunning = True

    # ------------------------------------------------------------------------------
    # The function that runs the thread
    def run(self):

        # Redirect stdout to the progress signal
        saveStdOut = sys.stdout
        stream = self.EmittingStream()
        stream.textWritten.connect(self.progress.emit)
        sys.stdout = stream

        # Start the analysis
        try:
            if self.settings.CPUCount > 1:
                self.externalRay = self.isRayRunning()

            self.started.emit("RUNNING")

            if self.settings.CPUCount > 1:
                sd.planarDICLocal(
                    self.settings,
                    self.resultsFile,
                    externalRay=self.externalRay,
                    guiThread=None,
                )
            else:
                sd.planarDICLocal(
                    self.settings, self.resultsFile, externalRay=False, guiThread=self
                )
            self.finished.emit("FINISHED")
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            self.finished.emit(f"Exception in thread: {e}")
        finally:
            sys.stdout = saveStdOut
            self._isRunning = False

    class EmittingStream(QObject):
        """Class to capture print statements and emit them as signals"""

        textWritten = pyqtSignal(str)

        # ------------------------------------------------------------------------------
        # Method to write text to the signal
        def write(self, text):
            if text:
                self.textWritten.emit(str(text))

        # ------------------------------------------------------------------------------
        # Method to flush the stream (not used here)
        def flush(self):
            pass  # Needed for compatibility

    # ------------------------------------------------------------------------------
    # Function to stop the thread
    def stop(self):
        # Implement a check in planarDICLocal() to read self._isRunning and abort gracefully.
        if self.settings.CPUCount <= 1:
            self._isRunning = False

        # If running in parallel, we cannot stop immediately, but we can shutdown ray
        else:
            sd._safeRayShutdown_(
                externalRay=self.externalRay, debugLevel=self.settings.DebugLevel
            )

    # ------------------------------------------------------------------------------
    # Function to check if the thread is running
    def isRunning(self):
        return self._isRunning

    # ------------------------------------------------------------------------------
    # Check if a ray server is already running on the local machine
    def isRayRunning(self, address="127.0.0.1", port=6379, timeout=1):
        try:
            with socket.create_connection((address, port), timeout=timeout):
                return True
        except OSError:
            return False

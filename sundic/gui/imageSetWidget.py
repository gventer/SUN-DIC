import os
import natsort as ns

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QListView, QGroupBox, QSpacerItem, QSizePolicy, QToolButton,
    QFileDialog, QLayout, QFrame
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem

import sundic.settings as sdset
from sundic.gui.validators import ClampingIntValidator, OddNumberValidator


class ImageSetUI(QWidget):
    """ Class for the image selection UI: Defines the layout and widgets for 
    the image selection tab
    """

    # ------------------------------------------------------------------------------
    # Initialize the image selection UI
    def __init__(self, parent):

        super().__init__(parent)
        self.parent = parent

        # Set the layout for this widget
        verticalLayout = QVBoxLayout(self)
        verticalLayout.setContentsMargins(20, 20, 20, 20)
        horizontalLayout = QHBoxLayout()

        # The image folder label and input
        folderLab = QLabel(self)
        sizePolicy = QSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        folderLab.setSizePolicy(sizePolicy)
        folderLab.setText("Folder:")
        horizontalLayout.addWidget(folderLab)

        self.folderDisp = QLabel(self)
        self.folderDisp.setText("PATH to Images...")
        self.folderDisp.setFrameShape(QFrame.Shape.Panel)
        self.folderDisp.setFrameShadow(QFrame.Shadow.Plain)
        self.folderDisp.setToolTip(
            "The folder containing the image set to analyze.")
        horizontalLayout.addWidget(self.folderDisp)

        self.selFolderBut = QPushButton(self)
        self.selFolderBut.setSizePolicy(sizePolicy)
        self.selFolderBut.setText("Select Directory")
        horizontalLayout.addWidget(self.selFolderBut)

        verticalLayout.addLayout(horizontalLayout)
        spacerItemV = QSpacerItem(
            20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        verticalLayout.addItem(spacerItemV)

        horizontalLayout_2 = QHBoxLayout()
        gridLayout = QGridLayout()

        # The start number input and label
        startLab = QLabel(self)
        startLab.setText("Start:")
        gridLayout.addWidget(startLab, 0, 0, 1, 1)

        self.startIn = QLineEdit(self)
        startValidator = ClampingIntValidator()
        startValidator.setBottom(1)
        self.startIn.setValidator(startValidator)
        self.startIn.setToolTip("""The image number of the first image in the set. 
Starts from 1.""")
        gridLayout.addWidget(self.startIn, 0, 1, 1, 1)

        # The end number input and label
        endLab = QLabel(self)
        endLab.setText("End:")
        gridLayout.addWidget(endLab, 1, 0, 1, 1)

        self.endIn = QLineEdit(self)
        endValidator = ClampingIntValidator()
        endValidator.setBottom(2)
        self.endIn.setValidator(endValidator)
        self.endIn.setText("2")
        self.endIn.setToolTip("The image number of the last image in the set.")
        gridLayout.addWidget(self.endIn, 1, 1, 1, 1)

        # The increment input and label
        incLab = QLabel(self)
        incLab.setText("Increment:")
        gridLayout.addWidget(incLab, 2, 0, 1, 1)

        self.incIn = QLineEdit(self)
        incValidator = ClampingIntValidator()
        incValidator.setBottom(1)
        self.incIn.setValidator(incValidator)
        self.incIn.setToolTip(
            "The increment between images to use in the analysis.")
        gridLayout.addWidget(self.incIn, 2, 1, 1, 1)

        spacerItemH = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        gridLayout.addItem(spacerItemH, 0, 2, 1, 1)

        # Set the maximum button
        self.setMax = QToolButton(self)
        self.setMax.setText("Set Max")
        gridLayout.addWidget(self.setMax, 1, 2, 1, 1)

        horizontalLayout_2.addLayout(gridLayout)

        # Images label
        imagesLab = QLabel(self)
        imagesLab.setText("Images:")
        horizontalLayout_2.addWidget(
            imagesLab, 0, QtCore.Qt.AlignmentFlag.AlignTop)

        # The image display list
        self.imageModel = QStandardItemModel()
        self.dispImages = QListView(self)
        self.dispImages.setFrameShape(QFrame.Shape.Panel)
        self.dispImages.setFrameShadow(QFrame.Shadow.Plain)
        self.dispImages.setModel(self.imageModel)
        self.dispImages.setToolTip(
            "The currently selected image set to analyze.")
        row_height = imagesLab.sizeHint().height()
        self.dispImages.setMaximumSize(
            QtCore.QSize(16777215, 5*row_height + 5))
        self.dispImages.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        horizontalLayout_2.addWidget(
            self.dispImages, 0, QtCore.Qt.AlignmentFlag.AlignTop)

        verticalLayout.addLayout(horizontalLayout_2)

        spacerItemV2 = QSpacerItem(
            20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        verticalLayout.addItem(spacerItemV2)

        # The group Box for advanced settings
        groupBox = QGroupBox(self)
        groupBox.setTitle("Advanced Settings")
        groupBox.setEnabled(True)
        sizePolicy = QSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            groupBox.sizePolicy().hasHeightForWidth())
        groupBox.setSizePolicy(sizePolicy)
        groupBox.setStyleSheet("QGroupBox { background-color: white; }")
        groupBox.setMaximumSize(QtCore.QSize(16777215, 5*row_height))
        gridLayoutWidget = QWidget(groupBox)

        gridLayoutWidget.setGeometry(QtCore.QRect(0, 10, 671, 81))
        gridLayoutWidget.setContentsMargins(10, 10, 10, 10)
        gridLayout_2 = QGridLayout(gridLayoutWidget)
        gridLayout_2.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        gridLayout_2.setContentsMargins(0, 10, 0, 0)

        spacerItem3 = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        gridLayout_2.addItem(spacerItem3, 0, 2, 1, 1)

        # The gaussian blur input and label
        gausLab = QLabel(gridLayoutWidget)
        gausLab.setText("Gaussian Blur:")
        gridLayout_2.addWidget(gausLab, 0, 0, 1, 1)

        self.gausIn = QLineEdit(gridLayoutWidget)
        gaussInValidator = OddNumberValidator(0, None)
        self.gausIn.setValidator(gaussInValidator)
        self.gausIn.setToolTip("""The size of the Gaussian blur to apply to the images. 
Must be an odd number larger than or equal to 0.""")
        gridLayout_2.addWidget(self.gausIn, 0, 1, 1, 1)

        # The background/cutoff input and label
        backLab = QLabel(gridLayoutWidget)
        backLab.setText("Background/Cutoff:")
        gridLayout_2.addWidget(backLab, 1, 0, 1, 1)

        self.backIn = QLineEdit(gridLayoutWidget)
        backValidator = ClampingIntValidator()
        backValidator.setBottom(0)
        backValidator.setTop(255)
        self.backIn.setValidator(backValidator)
        self.backIn.setToolTip("""Cutoff value to detect all black background in an image.
This value will be used to detect all black (< Cutoff) areas in the image.
This is useful for automatically removing unwanted areas from the image,
eg a hole in the sample. However, the background MUST be black.
Must be an integer between 0 and 255.""")
        gridLayout_2.addWidget(self.backIn, 1, 1, 1, 1)

        verticalLayout.addWidget(groupBox)

        spacerItemV1 = QSpacerItem(
            10, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        verticalLayout.addItem(spacerItemV1)

        # Set defaults button
        self.defaultsBut = QPushButton(self)
        self.defaultsBut.setText("Set Defaults (For this Panel Only)")
        verticalLayout.addWidget(self.defaultsBut)

        spacerItemV3 = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        verticalLayout.addItem(spacerItemV3)

        # Set the connections here
        self.defaultsBut.clicked.connect(self.setDefaults)
        self.selFolderBut.clicked.connect(self.openImageSetFolder)
        self.setMax.clicked.connect(self.setMaxTargetImage)

        # Connecting the input fields to the changedImageSet method to save the user input
        self.startIn.editingFinished.connect(self.changedImageSet)
        self.endIn.editingFinished.connect(self.changedImageSet)
        self.incIn.editingFinished.connect(self.changedImageSet)
        self.gausIn.editingFinished.connect(self.changedImageSet)
        self.backIn.editingFinished.connect(self.changedImageSet)

    # ------------------------------------------------------------------------------
    # Function to get the data from the settings UI and set it in the settings object
    def getData(self, settings):
        settings.ImageFolder = self.folderDisp.text()
        settings.DatumImage = int(self.startIn.text()) - 1
        settings.TargetImage = int(self.endIn.text()) - 1
        settings.Increment = int(self.incIn.text())
        settings.GaussianBlurSize = int(self.gausIn.text())
        settings.BackgroundCutoff = int(self.backIn.text())

    # ------------------------------------------------------------------------------
    # Function to get the data from this class and store it in the settings object
    def setData(self, settings):

        self.blockSignals(True)

        numImages = self.getNumImages()
        self.folderDisp.setText(settings.ImageFolder)
        self.startIn.setText(str(settings.DatumImage + 1))
        if settings.TargetImage == -1:
            numImages = max(numImages, 2)
            self.endIn.setText(str(numImages))
        else:
            self.endIn.setText(str(settings.TargetImage + 1))
        self.incIn.setText(str(settings.Increment))
        self.gausIn.setText(str(settings.GaussianBlurSize))
        self.backIn.setText(str(settings.BackgroundCutoff))
        self.updateImageList()

        self.blockSignals(False)

    # ------------------------------------------------------------------------------
    # Function that is called to indicate that the data was changed by the user
    def updateImageList(self):

        self.blockSignals(True)

        try:
            # Get all files in the image folder
            files = os.listdir(self.folderDisp.text())

            # Do a natural sort on the filenames and display them in the image list
            files = ns.os_sorted(files)
        except Exception as e:
            files = None

        # Setup the ItemModel from the start, end and increment values
        self.imageModel.clear()
        start = int(self.startIn.text()) - 1
        end = None
        inc = int(self.incIn.text())

        # Get the files to show and add to the model
        if files is not None:
            files = files[start:end:inc]
            for f in files:
                self.imageModel.appendRow(QStandardItem(f))

            # Set the target image value based on the available files
            self.endIn.setText(str(len(files)))
        else:
            self.endIn.setText("2")

        self.blockSignals(False)

    # ------------------------------------------------------------------------------
    # Function that is called to indicate that the data was changed by the user
    def changedImageSet(self):
        self.parent.savedFlag = False
        self.parent.updateWindowTitle()
        self.updateImageList()
        self.getData(self.parent.settings)

    # ------------------------------------------------------------------------------
    # Function that is called to indicate that the data was changed by the user
    def setDefaults(self):
        defSettings = sdset.Settings()
        self.setData(defSettings)
        self.changedImageSet()

    # ------------------------------------------------------------------------------
    # Function to open a file dialog to select the image folder
    def openImageSetFolder(self):

        # Open a file dialog to select the image folder
        options = QFileDialog.Option.ShowDirsOnly
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", "", options=options)

        # If a directory was selected, update the folder display and image list
        if directory:
            self.folderDisp.setText(directory)
            self.updateImageList()
            self.changedImageSet()

    # ------------------------------------------------------------------------------
    # Function to get the number of images in the selected folder
    def getNumImages(self):
        try:
            files = os.listdir(self.folderDisp.text())
            return len(files)
        except Exception as e:
            return 0

    # ------------------------------------------------------------------------------
    # Function to set the target image to the maximum available in the folder
    def setMaxTargetImage(self):

        self.blockSignals(True)

        numImages = self.getNumImages()
        self.endIn.setText(str(numImages))

        self.updateImageList()
        self.changedImageSet()

        self.blockSignals(False)

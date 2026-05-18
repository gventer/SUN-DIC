import os
import natsort as ns
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QSpacerItem, QSizePolicy, QGraphicsPixmapItem, QGraphicsView,
    QGraphicsScene, QGraphicsRectItem, QApplication, QFrame,
    QPushButton, QCheckBox, QFileDialog
)
from PyQt6.QtGui import QImage, QImageReader
from PIL import Image
from PyQt6.QtGui import QPixmap, QPen, QColor, QBrush, QCursor
from PyQt6.QtCore import pyqtSignal, Qt, QPointF, QRectF

from sundic.gui.validators import ClampingIntValidator
from sundic.sundic import readImage, _setupROI_, _setupSubSets_, _loadMask_, _buildActiveSubsetsMask_

class ROIDefUI(QWidget):
    """ Class for the ROI definition UI: Defines the layout and widgets for the 
        ROI definition tab
    """

    # ------------------------------------------------------------------------------
    # Initialize the image selection UI
    def __init__(self, parent):

        super().__init__(parent)

        self._loadingData = False # Guard varaible to deal with sync 
                                # between maskFile textfield

        # Set the class variables
        self.parent = parent

        verticalLayout = QVBoxLayout(self)
        verticalLayout.setContentsMargins(20, 20, 20, 20)
        horizontalLayout = QHBoxLayout()

        verticalLayout.addLayout(horizontalLayout)

        gridLayout = QGridLayout()
        gridLayout.setHorizontalSpacing(20)
        gridLayout.setVerticalSpacing(10)

        # Top left x label and input
        self.topLeftxDisp = QLabel(self)
        self.topLeftxDisp.setText("Top Left x:")
        gridLayout.addWidget(self.topLeftxDisp, 0, 0, 1, 1)

        self.xIn = QLineEdit(self)
        xInValidator = ClampingIntValidator()
        xInValidator.setBottom(0)
        self.xIn.setValidator(xInValidator)
        self.xIn.setToolTip("Top left x coordinate of the ROI.")
        gridLayout.addWidget(self.xIn, 0, 1, 1, 1)

        # Top left y label and input
        self.topLeftyDisp = QLabel(self)
        self.topLeftyDisp.setText("Top Left y:")
        gridLayout.addWidget(self.topLeftyDisp, 1, 0, 1, 1)

        self.yIn = QLineEdit(self)
        yInValidator = ClampingIntValidator()
        yInValidator.setBottom(0)
        self.yIn.setValidator(yInValidator)
        self.yIn.setToolTip("Top left y coordinate of the ROI.")
        gridLayout.addWidget(self.yIn, 1, 1, 1, 1)

        # Width label and input
        self.widthDisp = QLabel(self)
        self.widthDisp.setText("Width:")
        gridLayout.addWidget(self.widthDisp, 0, 2, 1, 1)

        self.widthIn = QLineEdit(self)
        widthValidator = ClampingIntValidator()
        widthValidator.setBottom(1)
        self.widthIn.setValidator(widthValidator)
        self.widthIn.setToolTip("Width of the ROI.")
        gridLayout.addWidget(self.widthIn, 0, 3, 1, 1)

        # Height label and input
        self.heightDisp = QLabel(self)
        self.heightDisp.setText("Height:")
        gridLayout.addWidget(self.heightDisp, 1, 2, 1, 1)

        self.heightIn = QLineEdit(self)
        heightValidator = ClampingIntValidator()
        heightValidator.setBottom(1)
        self.heightIn.setValidator(heightValidator)
        self.heightIn.setToolTip("Height of the ROI.")
        gridLayout.addWidget(self.heightIn, 1, 3, 1, 1)

        self.useMaskCheck = QCheckBox("Use ROI binary mask", self)
        self.useMaskCheck.setToolTip(
            "Use a binary mask image to define active subset centres.\n"
            "White pixels are analyzed, black pixels are ignored."
        )
        gridLayout.addWidget(self.useMaskCheck, 2, 0, 1, 2)

        self.maskFileDisp = QLabel("Mask file:", self)
        gridLayout.addWidget(self.maskFileDisp, 3, 0, 1, 1)

        self.maskFileIn = QLineEdit(self)
        self.maskFileIn.setToolTip(
            "Path to a binary mask image with the same size as the datum image."
        )
        gridLayout.addWidget(self.maskFileIn, 3, 1, 1, 2)

        self.maskBrowseBut = QPushButton("Browse...", self)
        gridLayout.addWidget(self.maskBrowseBut, 3, 3, 1, 1)        

        verticalLayout.addLayout(gridLayout)

        spacerItem = QSpacerItem(
            10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        verticalLayout.addItem(spacerItem)

        # Add the photo viewer for the ROI selection
        self.roiViewer = PhotoViewer(self)
        self.roiViewer.setFrameShape(QFrame.Shape.Panel)
        self.roiViewer.setFrameShadow(QFrame.Shadow.Plain)
        self.roiViewer.setToolTip("""Left Click+Drag to select the ROI.
Left Click+Shift+Drag to pan the image.
Use the mouse wheel to zoom in and out.
Red rectangle shows the selected ROI.
Green points show the active subset 
centers defined by the mask (if enabled).""")
        verticalLayout.addWidget(self.roiViewer)

        # Add a label to show the coordinates of the mouse
        self.labelCoords = QLabel(self)
        self.labelCoords.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        verticalLayout.addWidget(self.labelCoords)

        # Add connections here
        self.roiViewer.coordinatesChanged.connect(self.handleCoords)
        self.roiViewer.rectDrawn.connect(self.saveRect)

        self.xIn.editingFinished.connect(self.enterManualROI)
        self.yIn.editingFinished.connect(self.enterManualROI)
        self.widthIn.editingFinished.connect(self.enterManualROI)
        self.heightIn.editingFinished.connect(self.enterManualROI)

        self.useMaskCheck.toggled.connect(self.toggleMaskControls)
        self.maskBrowseBut.clicked.connect(self.browseMaskFile)
        self.useMaskCheck.toggled.connect(self.updateMaskPreview)
        self.maskFileIn.editingFinished.connect(self.updateMaskPreview)
        self.useMaskCheck.toggled.connect(self.changedMask)
        self.maskFileIn.editingFinished.connect(self.changedMask)
        self.toggleMaskControls(False)



    # ------------------------------------------------------------------------------
    # Function that is called to get data from this widget and store it in the settings object
    def getData(self, settings):
        xPos = int(self.xIn.text())
        yPos = int(self.yIn.text())
        width = int(self.widthIn.text())
        height = int(self.heightIn.text())
        settings.ROI = [xPos, yPos, width, height]
        if self.useMaskCheck.isChecked():
            settings.MaskFile = self.maskFileIn.text().strip()
        else:
            settings.MaskFile = ""

    # ------------------------------------------------------------------------------
    # Function that is called to set the data for this object
    def setData(self, settings):
        self._loadingData = True
        try:
            # Get the ROI definition
            xPos, yPos, width, height = settings.ROI
            self.xIn.setText(str(xPos))
            self.yIn.setText(str(yPos))
            self.widthIn.setText(str(width))
            self.heightIn.setText(str(height))
            hasMask = isinstance(settings.MaskFile, str) and len(settings.MaskFile.strip()) > 0
            fileName = settings.MaskFile.strip() if hasMask else "None"

            self.useMaskCheck.setChecked(hasMask)
            if hasMask:
                self.maskFileIn.setText(fileName)
            else:
                self.maskFileIn.setText("")


            try:
                imageFolder = self.parent.settings.ImageFolder
                files = os.listdir(imageFolder)
                files = ns.os_sorted(files)
                roiImage = files[self.parent.settings.DatumImage]
                imagePath = os.path.join(imageFolder, roiImage)

                pixmap = QPixmap()

                # First try Qt image loading
                reader = QImageReader(imagePath)
                image = reader.read()
                if not image.isNull():
                    pixmap = QPixmap.fromImage(image)

                # Fallback to Pillow if Qt failed
                if pixmap.isNull():
                    pilImage = Image.open(imagePath).convert("RGBA")
                    data = pilImage.tobytes("raw", "RGBA")
                    qimage = QImage(
                        data,
                        pilImage.width,
                        pilImage.height,
                        pilImage.width * 4,
                        QImage.Format.Format_RGBA8888
                    )
                    pixmap = QPixmap.fromImage(qimage.copy())

                self.roiViewer.setPhoto(pixmap)
                self.roiViewer.setRect(xPos, yPos, width, height)
                self.updateMaskPreview()

            except Exception as e:
                print("Error setting ROI data:", e)
                import traceback
                traceback.print_exc()
                self.roiViewer.setPhoto(QPixmap())   
        finally:
            self._loadingData = False          

    # ------------------------------------------------------------------------------
    # Function that updates the display of coordinates
    def handleCoords(self, point):
        if not point.isNull():
            self.labelCoords.setText(f'({int(point.x())}, {int(point.y())})')
        else:
            self.labelCoords.setText(" ")

    # ------------------------------------------------------------------------------
    # Save the rectangle drawn on the image based on signal from PhotoViewer
    def saveRect(self, x, y, width, height):
        self.xIn.setText(str(int(x)))
        self.yIn.setText(str(int(y)))
        self.widthIn.setText(str(int(width)))
        self.heightIn.setText(str(int(height)))
        self.changedROI()

        # Update the settings object and main Window status
        self.getData(self.parent.settings)
        self.parent.savedFlag = False
        self.parent.updateWindowTitle()

        self.updateMaskPreview()

    # ------------------------------------------------------------------------------
    # Function that is called when the user changes the ROI definition manually
    def enterManualROI(self):

        x = int(self.xIn.text())
        y = int(self.yIn.text())
        width = int(self.widthIn.text())
        height = int(self.heightIn.text())

        rect = QRectF(x, y, width, height)
        rect = rect.intersected(self.roiViewer.photo.boundingRect())
        if rect.isNull():
            rect = QRectF(0, 0, 1, 1)

        self.xIn.setText(str(int(rect.x())))
        self.yIn.setText(str(int(rect.y())))
        self.widthIn.setText(str(int(rect.width())))
        self.heightIn.setText(str(int(rect.height())))

        pen = QPen()
        pen.setColor(Qt.GlobalColor.red)
        pen.setWidth(3)

        brush = QBrush()
        brush.setColor(QColor(255, 0, 0, 80))
        brush.setStyle(Qt.BrushStyle.SolidPattern)

        if self.roiViewer._previewRect:
            self.roiViewer.scene.removeItem(self.roiViewer._previewRect)
            self.roiViewer._previewRect = None
        if self.roiViewer._finalRect:
            self.roiViewer.scene.removeItem(self.roiViewer._finalRect)
            self.roiViewer._finalRect = None

        self.roiViewer._finalRect = QGraphicsRectItem(rect)
        self.roiViewer._finalRect.setPen(pen)
        self.roiViewer._finalRect.setBrush(brush)
        self.roiViewer.scene.addItem(self.roiViewer._finalRect)

        self.updateMaskPreview()
        self.changedROI()

    # ------------------------------------------------------------------------------
    # Helper function to update the ROI definition when the user changes the values manually
    def changedROI(self):
        """Save ROI changes immediately to the settings object."""
        if self._loadingData:
            return
        try:
            self.getData(self.parent.settings)
            self.parent.savedFlag = False
            self.parent.updateWindowTitle()
        except ValueError:
            # Ignore incomplete edits until all fields contain valid integers
            pass

    # ------------------------------------------------------------------------------
    # Helper function to update the mask preview when the user changes the mask file 
    # or toggles the use mask option
    def changedMask(self):
        """Save mask changes immediately to the settings object."""
        if self._loadingData:
            return
        self.getData(self.parent.settings)
        self.parent.savedFlag = False
        self.parent.updateWindowTitle()


    # ------------------------------------------------------------------------------
    # Helper function to toggle the mask file input and browse button based on the 
    # state of the checkbox
    def toggleMaskControls(self, checked):
        """Enable or disable the mask file input and browse button based on the state 
        of the "Use ROI binary mask" checkbox. When the checkbox is unchecked, the 
        mask file input is cleared and both the input and browse button are disabled.

        Parameters:
            checked (bool): The state of the checkbox (True if checked, False if unchecked)
        """
        self.maskFileIn.setEnabled(checked)
        self.maskBrowseBut.setEnabled(checked)


    # ------------------------------------------------------------------------------
    # Helper function to open a file dialog for selecting the mask file and update 
    # the mask file input with the selected file path
    def browseMaskFile(self):
        """Open a file dialog for selecting a binary mask image file and update the mask
        file input with the selected file path. The file dialog filters for common image
        formats. If a file is selected, the "Use ROI binary mask" checkbox is checked
        and the mask preview is updated.
        """
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select ROI Mask",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if fileName:
            self.maskFileIn.setText(fileName)
            self.useMaskCheck.setChecked(True)
            self.updateMaskPreview()
            self.changedMask()

    # -----------------------------------------------------------------------------
    # Helper function to update the mask preview points on the image based on the 
    # current ROI definition and mask file
    def updateMaskPreview(self):
        """Update the mask preview points on the image based on the current ROI definition and mask file.
        """
        try:
            imageFolder = self.parent.settings.ImageFolder
            files = ns.os_sorted(os.listdir(imageFolder))
            roiImage = files[self.parent.settings.DatumImage]
            firstImagePath = os.path.join(imageFolder, roiImage)

            settings = self.parent.settings
            roiVals = [
                int(self.xIn.text()),
                int(self.yIn.text()),
                int(self.widthIn.text()),
                int(self.heightIn.text())
            ]

            img0 = readImage(firstImagePath, normalize8Bit=True)
            ROI = _setupROI_(roiVals, firstImagePath)
            subSetPnts = _setupSubSets_(
                self.parent.settings.SubsetSize,
                self.parent.settings.StepSize,
                self.parent.settings.ShapeFunctions,
                ROI,
                firstImagePath, debugLevel=0
            )

            activeSubsets = np.ones(subSetPnts.shape[:2], dtype=bool)
            if self.useMaskCheck.isChecked():
                maskFile = self.maskFileIn.text().strip()
                if len(maskFile) > 0:
                    roiMask = _loadMask_(maskFile, img0.shape)
                    activeSubsets = _buildActiveSubsetsMask_(subSetPnts, roiMask)

            x = subSetPnts[:, :, 0][activeSubsets]
            y = subSetPnts[:, :, 1][activeSubsets]
            self.roiViewer.setMaskPreviewPoints(x, y)

        except Exception as e:
            print("Error updating mask preview:", e)
            self.roiViewer.clearMaskPreview()


        
class PhotoViewer(QGraphicsView):
    """ Class for the photo viewer used in the ROI definition tab
    """

    coordinatesChanged = pyqtSignal(QPointF)
    rectDrawn = pyqtSignal(float, float, float, float)
    SCALE_FACTOR = 1.25

    # ------------------------------------------------------------------------------
    # Initialize the photo viewer
    def __init__(self, parent=None):

        super().__init__(parent)

        self.zoomLevel = 0
        self.hasPhoto = False
        self._drawing = False
        self._startPoint = None
        self._previewRect = None
        self._finalRect = None
        self.currentScale = 1.0

        self._maskPreviewItems = []

        self.scene = QGraphicsScene(self)
        self.photo = QGraphicsPixmapItem()
        self.scene.addItem(self.photo)
        self.setScene(self.scene)

        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                        QSizePolicy.Policy.Expanding)

    # ------------------------------------------------------------------------------
    # Function to reset the view
    def resetView(self):
        rect = self.photo.boundingRect()
        if rect.isNull():
            return

        self.setSceneRect(rect)
        self.resetTransform()

        if self.viewport().width() > 0 and self.viewport().height() > 0:
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            if self.currentScale != 1.0:
                self.scale(self.currentScale, self.currentScale)

    # ------------------------------------------------------------------------------
    # Function to set the photo to be displayed
    def setPhoto(self, pixmap=None):
        self.zoomLevel = 0
        self.currentScale = 1.0

        if pixmap is not None and not pixmap.isNull():
            self.hasPhoto = True
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.photo.setPixmap(pixmap)
            self.maxWidth = pixmap.width()
            self.maxHeight = pixmap.height()
            self.scene.setSceneRect(self.photo.boundingRect())
        else:
            self.hasPhoto = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.photo.setPixmap(QPixmap())
            self.scene.setSceneRect(QRectF())

        if self._previewRect:
            self.scene.removeItem(self._previewRect)
            self._previewRect = None

        if self._finalRect:
            self.scene.removeItem(self._finalRect)
            self._finalRect = None

        self.clearMaskPreview()
        self.resetView()
        self.viewport().update()

    # ------------------------------------------------------------------------------
    # Function to zoom in or out
    def zoom(self, step):
        if not self.hasPhoto:
            return

        step = int(step)
        self.zoomLevel += step
        scaleFactor = self.SCALE_FACTOR ** step if step > 0 else 1 / \
            (self.SCALE_FACTOR ** abs(step))
        self.currentScale *= scaleFactor
        self.scale(scaleFactor, scaleFactor)

    # ------------------------------------------------------------------------------
    # Function to handle the mouse wheel event (zooming in this case)
    def wheelEvent(self, event):
        if not self.hasPhoto:
            super().wheelEvent(event)
            return

        delta = event.angleDelta().y()
        if delta:
            self.zoom(1 if delta > 0 else -1)

    # ------------------------------------------------------------------------------
    # Function to resize the view
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resetView()

    # ------------------------------------------------------------------------------
    # Function to handle mouse press events
    def mousePressEvent(self, event):

        # Only proceed if we have a photo
        if not self.hasPhoto:
            super().mousePressEvent(event)
            return

        # Enable panning with Shift + Left Click
        if (event.button() == Qt.MouseButton.LeftButton and
                QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier):
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            super().mousePressEvent(event)

        # Enable drawing the ROI with Left Click
        elif event.button() == Qt.MouseButton.LeftButton:
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            scenePos = self.mapToScene(event.pos())
            self._startPoint = scenePos
            self._drawing = True

            if self._previewRect:
                self.scene.removeItem(self._previewRect)
                self._previewRect = None

        # Ignore the rest
        else:
            super().mousePressEvent(event)

    # ------------------------------------------------------------------------------
    # Function to handle mouse move events
    def mouseMoveEvent(self, event):

        # Only proceed if we have a photo
        if not self.hasPhoto:
            super().mouseMoveEvent(event)
            return

        # When dragging (panning)
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            super().mouseMoveEvent(event)

        # When drawing the rectangle
        elif self._drawing:
            scenePos = self.mapToScene(event.pos())

            if self._previewRect:
                self.scene.removeItem(self._previewRect)
                self._previewRect = None

            rect = QRectF(self._startPoint, scenePos).normalized()
            rect = rect.intersected(self.photo.boundingRect())
            if rect.isNull():
                rect = QRectF(0, 0, 1, 1)

            pen = QPen(Qt.GlobalColor.red, 2)
            brush = QBrush(QColor(255, 0, 0, 80))
            self._previewRect = QGraphicsRectItem(rect)
            self._previewRect.setPen(pen)
            self._previewRect.setBrush(brush)
            self.scene.addItem(self._previewRect)

        # Ignore the rest
        else:
            super().mouseMoveEvent(event)

        # Update the coordinates to display as the mouse is moved
        self.updateCoordinates(event.pos())

    # ------------------------------------------------------------------------------
    # Function to handle mouse release events
    def mouseReleaseEvent(self, event):

        # Only proceed if we have a photo
        if not self.hasPhoto:
            super().mouseReleaseEvent(event)
            return

        # End of panning
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            super().mouseReleaseEvent(event)

        # End of drawing the rectangle
        elif self._drawing and event.button() == Qt.MouseButton.LeftButton:
            scenePos = self.mapToScene(event.pos())
            rect = QRectF(self._startPoint, scenePos).normalized()

            if self._previewRect:
                self.scene.removeItem(self._previewRect)
                self._previewRect = None

            if self._finalRect:
                self.scene.removeItem(self._finalRect)
                self._finalRect = None

            pen = QPen(Qt.GlobalColor.red, 3)
            brush = QBrush(QColor(255, 0, 0, 80))

            # Adjust the rectangle to be within image bounds
            rect = rect.intersected(self.photo.boundingRect())
            if rect.isNull():
                rect = QRectF(0, 0, 1, 1)

            self.rectDrawn.emit(rect.x(), rect.y(),
                                rect.width(), rect.height())

            self._finalRect = QGraphicsRectItem(rect)
            self._finalRect.setPen(pen)
            self._finalRect.setBrush(brush)
            self.scene.addItem(self._finalRect)

            self._drawing = False

        # Ignore the rest
        else:
            super().mouseReleaseEvent(event)

        # Reset the cursor
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # ------------------------------------------------------------------------------
    # Function to update the coordinates display
    def updateCoordinates(self, pos=None):

        # Check if we have a position to display
        if pos is None:
            pos = self.mapFromGlobal(QCursor.pos())

        # Convert the position to scene coordinates
        scene_point = self.mapToScene(pos)
        item_point = self.photo.mapFromScene(scene_point)

        # If the point is inside the photo rectangle emit the point
        if self.photo.contains(item_point):
            self.coordinatesChanged.emit(scene_point)
        else:
            self.coordinatesChanged.emit(QPointF())

    # ------------------------------------------------------------------------------
    # Function that is called when the mouse leaves the widget
    def leaveEvent(self, event):
        self.coordinatesChanged.emit(QPointF())
        super().leaveEvent(event)

    # ------------------------------------------------------------------------------
    # Function to set the rectangle on the image based on ROI definition
    def setRect(self, x, y, width, height):
        rect = QRectF(x, y, width, height)
        rect = rect.intersected(self.photo.boundingRect())
        if rect.isNull():
            rect = QRectF(0, 0, 1, 1)

        pen = QPen(Qt.GlobalColor.red)
        pen.setWidth(3)
        brush = QBrush(QColor(255, 0, 0, 80))
        brush.setStyle(Qt.BrushStyle.SolidPattern)

        if self._previewRect:
            self.scene.removeItem(self._previewRect)
            self._previewRect = None

        if self._finalRect:
            self.scene.removeItem(self._finalRect)
            self._finalRect = None
        self._finalRect = QGraphicsRectItem(rect)
        self._finalRect.setPen(pen)
        self._finalRect.setBrush(brush)
        self.scene.addItem(self._finalRect)

    # ------------------------------------------------------------------------------
    # Function to clear the mask preview items from the scene
    def clearMaskPreview(self):
        """Remove any existing mask preview items from the scene and clear the list of mask preview items.
        """
        for item in self._maskPreviewItems:
            self.scene.removeItem(item)

        self._maskPreviewItems = []

    # ------------------------------------------------------------------------------
    # Function to set the mask preview points on the image based on the coordinates 
    # of the active points defined by the mask
    def setMaskPreviewPoints(self, xCoords, yCoords):
        """ Add small green circles to the scene at the specified x and y 
            coordinates to preview the active points defined by the mask. Any existing 
            mask preview items are cleared before adding the new ones.

            Parameters:
                xCoords (array-like): An array of x coordinates for the active points defined by the mask.
                yCoords (array-like): An array of y coordinates for the active points defined by the mask.
        """
        self.clearMaskPreview()

        pen = QPen(Qt.GlobalColor.green)
        brush = QBrush(QColor(0, 255, 0, 180))

        for x, y in zip(xCoords, yCoords):
            item = self.scene.addEllipse(x - 1.5, y - 1.5, 3, 3, pen, brush)
            self._maskPreviewItems.append(item)

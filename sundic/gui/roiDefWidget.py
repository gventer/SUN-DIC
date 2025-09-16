import os
import natsort as ns

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QSpacerItem, QSizePolicy, QGraphicsPixmapItem, QGraphicsView,
    QGraphicsScene, QGraphicsRectItem, QApplication, QFrame
)
from PyQt6.QtGui import QPixmap, QPen, QColor, QBrush, QCursor
from PyQt6.QtCore import pyqtSignal, Qt, QPointF, QRectF

from sundic.gui.validators import ClampingIntValidator


class ROIDefUI(QWidget):
    """ Class for the ROI definition UI: Defines the layout and widgets for the 
        ROI definition tab
    """

    # ------------------------------------------------------------------------------
    # Initialize the image selection UI
    def __init__(self, parent):

        super().__init__(parent)

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

        verticalLayout.addLayout(gridLayout)

        spacerItem = QSpacerItem(
            10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        verticalLayout.addItem(spacerItem)

        # Add the photo viewer for the ROI selection
        self.roiViewer = PhotoViewer(self)
        self.roiViewer.setToolTip("""Left Click+Drag to select the ROI.
Left Click+Shift+Drag to pan the image.
Use the mouse wheel to zoom in and out.""")
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

    # ------------------------------------------------------------------------------
    # Function that is called to get data from this widget and store it in the settings object
    def getData(self, settings):
        xPos = int(self.xIn.text())
        yPos = int(self.yIn.text())
        width = int(self.widthIn.text())
        height = int(self.heightIn.text())
        settings.ROI = [xPos, yPos, width, height]

    # ------------------------------------------------------------------------------
    # Function that is called to set the data for this object
    def setData(self, settings):

        # Get the ROI definition
        xPos, yPos, width, height = settings.ROI
        self.xIn.setText(str(xPos))
        self.yIn.setText(str(yPos))
        self.widthIn.setText(str(width))
        self.heightIn.setText(str(height))

        # Get all files in the image folder and then the datum image
        try:
            imageFolder = self.parent.settings.ImageFolder
            files = os.listdir(imageFolder)
            files = ns.os_sorted(files)
            roiImage = files[self.parent.settings.DatumImage]

            # Setting the datum photo to be displayed
            self.roiViewer.setPhoto(
                QPixmap(os.path.join(imageFolder, roiImage)))

            # Set the rectangle on the image based on the ROI definition
            self.roiViewer.setRect(xPos, yPos, width, height)
        except Exception as e:
            pass

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

        # Update the settings object and main Window status
        self.getData(self.parent.settings)
        self.parent.savedFlag = False
        self.parent.updateWindowTitle()

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
        if self.roiViewer._finalRect:
            self.roiViewer.scene.removeItem(self.roiViewer._finalRect)

        self.roiViewer._finalRect = QGraphicsRectItem(rect)
        self.roiViewer._finalRect.setPen(pen)
        self.roiViewer._finalRect.setBrush(brush)
        self.roiViewer.scene.addItem(self.roiViewer._finalRect)


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

        self.scene = QGraphicsScene(self)
        self.photo = QGraphicsPixmapItem()
        self.scene.addItem(self.photo)
        self.setScene(self.scene)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # ------------------------------------------------------------------------------
    # Function to reset the view
    def resetView(self, scale=1):

        rect = QRectF(self.photo.pixmap().rect())

        if not rect.isNull():
            self.setSceneRect(rect)
            unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
            self.resetTransform()
            self.scale(1 / unity.width(), 1 / unity.height())
            viewRect = self.viewport().rect()
            sceneRect = self.transform().mapRect(rect)
            factor = min(viewRect.width() / sceneRect.width(),
                         viewRect.height() / sceneRect.height())
            self.scale(factor, factor)
            self.centerOn(self.photo)

            # Apply user zoom
            self.scale(self.currentScale, self.currentScale)

    # ------------------------------------------------------------------------------
    # Function to set the photo to be displayed
    def setPhoto(self, pixmap=None):
        if pixmap is not None and not pixmap.isNull():
            self.hasPhoto = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.photo.setPixmap(pixmap)
            self.maxWidth = pixmap.width()
            self.maxHeight = pixmap.height()
        else:
            self.hasPhoto = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.photo.setPixmap(QPixmap())
        self.resetView(self.SCALE_FACTOR ** self.zoomLevel)

    # ------------------------------------------------------------------------------
    # Function to zoom in or out
    def zoom(self, step):
        step = int(step)
        self.zoomLevel += step
        scaleFactor = self.SCALE_FACTOR ** step if step > 0 else 1 / \
            (self.SCALE_FACTOR ** abs(step))
        self.currentScale *= scaleFactor
        self.scale(scaleFactor, scaleFactor)

    # ------------------------------------------------------------------------------
    # Function to handle the mouse wheel event (zooming in this case)
    def wheelEvent(self, event):
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
            rect = QRectF(self._startPoint, scenePos).normalized()
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
        point = self.mapToScene(pos)

        # If the point is inside the photo rectangle emit the point
        if self.photo.contains(point):
            self.coordinatesChanged.emit(point)

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

        if self._finalRect:
            self.scene.removeItem(self._finalRect)
        self._finalRect = QGraphicsRectItem(rect)
        self._finalRect.setPen(pen)
        self._finalRect.setBrush(brush)
        self.scene.addItem(self._finalRect)

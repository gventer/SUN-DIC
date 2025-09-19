from PyQt6 import QtWidgets, QtGui, QtCore


class LeftIconButton(QtWidgets.QPushButton):
    """ A QPushButton subclass that places the icon on the left side of the button
    """

    # -------------------------------------------------------------------------------
    def __init__(self, *args, stylesheet=None, **kwargs):
        """ Constructor - optional stylesheet argument
        """
        super().__init__(*args, **kwargs)

        if stylesheet is not None:
            self.setStyleSheet(stylesheet)

    # ------------------------------------------------------------------------------
    def sizeHint(self):
        """ Override the sizeHint method to calculate the size based on text and icon
        """

        # Get base size for text
        fontMetrics = QtGui.QFontMetrics(self.font())
        text = self.text()
        textWidth = fontMetrics.horizontalAdvance(text)
        textHeight = fontMetrics.height()

        # Get icon size
        iconSize = self.iconSize()
        iconWidth = iconSize.width()
        iconHeight = iconSize.height()

        # Padding between text and icon
        spacing = 10 if (not self.icon().isNull() and text) else 0

        # Calculate total width: text + spacing + icon + left/right margins
        width = textWidth + spacing + iconWidth + 10
        height = max(textHeight, iconHeight) + 10

        return QtCore.QSize(width, height)

    # ------------------------------------------------------------------------------
    def paintEvent(self, event):
        """ Override the paintEvent method to custom draw the button
        """

        # The painter and button style
        painter = QtGui.QPainter(self)
        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        self.style().drawControl(
            QtWidgets.QStyle.ControlElement.CE_PushButtonBevel,
            option, painter, self)

        # Draw text centered
        rect = self.rect()
        iconSize = self.iconSize()
        text = self.text()
        fontMetrics = QtGui.QFontMetrics(self.font())
        textRect = fontMetrics.boundingRect(
            rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)

        # Reserve space for icon on the left
        textRect.setRight(textRect.right()+iconSize.width()//2)
        textRect.setLeft(textRect.left()+iconSize.width()//2)
        painter.drawText(textRect, QtCore.Qt.AlignmentFlag.AlignLeft, text)

        # Draw icon on the left
        iconX = 5
        iconY = (rect.height() - iconSize.height()) // 2
        self.icon().paint(painter, iconX, iconY, iconSize.width(), iconSize.height())

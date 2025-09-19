from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QSpacerItem, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QPainter

import os


class AboutDialog(QDialog):
    """ Class for the About dialog
    """

    def __init__(self, parent=None, version="1.0.0"):

        super().__init__(parent)

        self.setWindowTitle("About SUN-DIC")
        self.setModal(True)

        scriptDir = os.path.dirname(os.path.abspath(__file__))
        iconsPath = os.path.join(scriptDir, "icons", "about.png")
        self.bgPixmap = QPixmap(iconsPath)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 18, 24, 18)

        # Title and version
        nameVersion = QLabel(
            f"<b><span style='font-size:18pt'>SUN-DIC</span></b> <span style='font-size:12pt'>v{version}</span>")
        nameVersion.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(nameVersion)

        # Project description from GitHub
        desc = QLabel(
            "SUN-DIC is an open-source Digital Image Correlation (DIC) software for analyzing material deformation using image sets. "
            "For more information, documentation, and source code, visit the project's GitHub page below."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Minimum)
        desc.setMaximumWidth(420)
        layout.addWidget(desc)

        # Spacer
        layout.addSpacerItem(QSpacerItem(
            10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Github repository link
        githubLink = QLabel(
            '<a href="https://github.com/gventer/SUN-DIC">https://github.com/gventer/SUN-DIC</a>'
        )
        githubLink.setAlignment(Qt.AlignmentFlag.AlignCenter)
        githubLink.setOpenExternalLinks(True)
        layout.addWidget(githubLink)

        # Copyright/license
        copyright = QLabel(
            '<span style="font-size:8pt; color:black;">&copy; 2023-2025 SUN-DIC contributors. '
            'Distributed under the MIT License.</span>'
        )
        copyright.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(copyright)

        # Close button
        closeBtn = QPushButton("Close")
        closeBtn.clicked.connect(self.accept)
        layout.addWidget(closeBtn)
        layout.setAlignment(closeBtn, Qt.AlignmentFlag.AlignCenter)

        # Auto-resize the dialog to fit content
        self.adjustSize()
        self.setMinimumWidth(380)   # Optional: ensure dialog is not too narrow

    # ------------------------------
    # Custom paint event to draw background image
    def paintEvent(self, event):

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Set opacity
        painter.setOpacity(0.2)

        # Lets scale the image to fit the dialog while maintaining aspect ratio
        pixmapRatio = self.bgPixmap.width() / self.bgPixmap.height()
        dialogRatio = self.width() / self.height()
        if pixmapRatio > dialogRatio:
            scaledWidth = self.width()
            scaledHeight = int(scaled_width / pixmap_ratio)
        else:
            scaledHeight = self.height()
            scaledWidth = int(scaledHeight * pixmapRatio)

        scaledPixmap = self.bgPixmap.scaled(
            scaledWidth, scaledHeight,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Draw the scaled image as background in the center
        x = (self.width() - scaledPixmap.width()) // 2
        y = (self.height() - scaledPixmap.height()) // 2
        painter.drawPixmap(x, y, scaledPixmap)

        # Don't forget to call the base class implementation to draw child widgets
        painter.setOpacity(1.0)
        super().paintEvent(event)

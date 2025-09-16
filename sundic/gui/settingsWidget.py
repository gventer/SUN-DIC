import sundic.util.datafile as dataFile
import sundic.settings as sdset
from sundic.gui.validators import OddNumberValidator, IntListValidator, ClampingIntValidator
from sundic.gui.validators import ClampingDblValidator
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QComboBox,
    QSpacerItem, QSizePolicy
)


class SettingsUI(QWidget):
    """ Class for the settings UI: Defines the layout and widgets for the settings tab
        in the main window. Also contains functions to get and set the data from the
        settings object.
    """

    # ------------------------------------------------------------------------------
    # Initialize the settings UI
    def __init__(self, parent):

        super().__init__(parent)

        # Set the class variables
        self.parent = parent

        # Set the layout for this widget
        verticalLayout = QVBoxLayout(self)
        verticalLayout.setContentsMargins(20, 20, 20, 20)

        # Set the grid layout for this widget
        gridLayout = QGridLayout()
        gridLayout.setSpacing(10)

        # Add the DIC Type label and combo box
        dicTypeLab = QLabel(self)
        dicTypeLab.setText("DIC Type:")
        gridLayout.addWidget(dicTypeLab, 0, 0, 1, 1)

        self.dicTypeBox = QComboBox(self)
        dicTypeBoxItems = ["Planar"]
        for item in dicTypeBoxItems:
            self.dicTypeBox.addItem(item)
        self.dicTypeBox.setToolTip("""The type of DIC analysis to be performed. 
Currently only planar is available.""")
        gridLayout.addWidget(self.dicTypeBox, 0, 1, 1, 1)

        # Add the shape function label and combo box
        shapeFuncLab = QLabel(self)
        shapeFuncLab.setText("Shape Function:")
        gridLayout.addWidget(shapeFuncLab, 1, 0, 1, 1)

        self.shapeFuncBox = QComboBox(self)
        shapeFuncBoxItems = ["Affine", "Quadratic"]
        for item in shapeFuncBoxItems:
            self.shapeFuncBox.addItem(item)
        self.shapeFuncBox.setToolTip(
            "Subset shape functions to use.")
        gridLayout.addWidget(self.shapeFuncBox, 1, 1, 1, 1)

        # Add the subset size input and label
        subsetSizeLab = QLabel(self)
        subsetSizeLab.setText("Subset Size:")
        gridLayout.addWidget(subsetSizeLab, 2, 0, 1, 1)

        self.subsetSizeIn = QLineEdit(self)
        subSetValidator = OddNumberValidator(5, None)
        self.subsetSizeIn.setValidator(subSetValidator)
        self.subsetSizeIn.setToolTip("""The subset size used for the DIC analysis.
Must be larger than or equal to 5 and odd.""")
        gridLayout.addWidget(self.subsetSizeIn, 2, 1, 1, 1)

        # Add the step size input and label
        stepSizeLab = QLabel(self)
        stepSizeLab.setText("Step Size:")
        gridLayout.addWidget(stepSizeLab, 3, 0, 1, 1)

        self.stepSizeIn = QLineEdit(self)
        intValidatorStepSize = ClampingIntValidator()
        intValidatorStepSize.setBottom(1)
        self.stepSizeIn.setValidator(intValidatorStepSize)
        self.stepSizeIn.setToolTip("""The step size used for the DIC analysis.
Must be larger than or equal to 1.""")
        gridLayout.addWidget(self.stepSizeIn, 3, 1, 1, 1)

        # Add the reference strategy input and label
        refStartLab = QLabel(self)
        refStartLab.setText("Reference Strategy:")
        gridLayout.addWidget(refStartLab, 4, 0, 1, 1)

        self.refBox = QComboBox(self)
        refBoxItems = ["Relative", "Absolute"]
        for item in refBoxItems:
            self.refBox.addItem(item)

        self.refBox.setToolTip("""The reference strategy to use. 
Absolute - The reference image is the first image. 
           Useful for small deformations.
Relative - The reference image is the previous image.
           Useful for large deformations.""")
        gridLayout.addWidget(self.refBox, 4, 1, 1, 1)

        # Add the algorithm type input and label
        algorLab = QLabel(self)
        algorLab.setText("Algorithm:")
        gridLayout.addWidget(algorLab, 0, 2, 1, 1)

        self.algoTypeBox = QComboBox(self)
        algoTypeBoxItems = ["IC-GN", "IC-LM", "Fast-IC-LM"]
        for item in algoTypeBoxItems:
            self.algoTypeBox.addItem(item)

        self.algoTypeBox.setToolTip("""The optimization algorithm to use. 
IC-GN      - Use the Inverse Compositional Gauss Newton algorithm.
IC-LM      - Use the Inverse Compositional Levenberg-Marquardt algorithm. 
Fast-IC-LM - Use a faster version of the IC-LM algorithm""")
        gridLayout.addWidget(self.algoTypeBox, 0, 3, 1, 1)

        # The starting points input and label
        startingPLab = QLabel(self)
        startingPLab.setText("Starting Points:")
        gridLayout.addWidget(startingPLab, 1, 2, 1, 1)

        self.startingPIn = QLineEdit(self)
        intValidatorStartP = ClampingIntValidator()
        intValidatorStartP.setBottom(1)
        self.startingPIn.setValidator(intValidatorStartP)
        self.startingPIn.setToolTip("""The number of starting points used for the DIC analysis. 
The total number of points will be the number of starting points squared.
Must be larger than or equal to 1.""")
        gridLayout.addWidget(self.startingPIn, 1, 3, 1, 1)

        # The convergence criteria input and label
        convergenceLab = QLabel(self)
        convergenceLab.setText("Convergence:")
        gridLayout.addWidget(convergenceLab, 2, 2, 1, 1)

        self.convergenceIn = QLineEdit(self)
        dblValidatorConv = ClampingDblValidator(0, 100)
        self.convergenceIn.setValidator(dblValidatorConv)
        self.convergenceIn.setToolTip("""The convergence threshold for the optimization algorithm. 
Must be larger than 0.""")
        gridLayout.addWidget(self.convergenceIn, 2, 3, 1, 1)

        # The zncc tolerance input and label
        znccTolLab = QLabel(self)
        znccTolLab.setText("ZNCC Tolerance:")
        gridLayout.addWidget(znccTolLab, 3, 2, 1, 1)

        self.znccTolIn = QLineEdit(self)
        dblValidatorZNCC = ClampingDblValidator(0, 100)
        self.znccTolIn.setValidator(dblValidatorZNCC)
        self.znccTolIn.setToolTip("""The ZNCC tolerance for convergence. 
Must be larger than 0 and less than or equal to 1.""")
        gridLayout.addWidget(self.znccTolIn, 3, 3, 1, 1)

        # The interpolation order input and label
        interpOrderLab = QLabel(self)
        interpOrderLab.setText("Interpolation Order:")
        gridLayout.addWidget(interpOrderLab, 4, 2, 1, 1)

        self.interpOrderIn = QLineEdit(self)
        self.interpOrderIn.setValidator(IntListValidator([1, 3, 5]))
        self.interpOrderIn.setToolTip("""The interpolation order to use. 
Must be 1, 3 or 5.""")
        gridLayout.addWidget(self.interpOrderIn, 4, 3, 1, 1)

        # The maximum iterations input and label
        maxIterLab = QLabel(self)
        maxIterLab.setText("Max Iterations:")
        gridLayout.addWidget(maxIterLab, 5, 2, 1, 1)

        self.maxIterIn = QLineEdit(self)
        intValidatorMaxIter = ClampingIntValidator()
        intValidatorMaxIter.setBottom(1)
        self.maxIterIn.setValidator(intValidatorMaxIter)
        self.maxIterIn.setToolTip(
            """The maximum number of iterations used for the optimization algorithm. 
Must be larger than or equal to 1.
The default value (50) is set conservatively high and should rarely be changed.""")
        gridLayout.addWidget(self.maxIterIn, 5, 3, 1, 1)

        # Add the grid layout to the vertical layout
        verticalLayout.addLayout(gridLayout)
        spacerItemV1 = QSpacerItem(
            10, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        verticalLayout.addItem(spacerItemV1)

        # Set defaults button
        self.defaultsBut = QPushButton(self)
        self.defaultsBut.setText("Set Defaults (For this Panel Only)")
        verticalLayout.addWidget(self.defaultsBut)

        # Add a spacer to push everything to the top
        spacerItemV = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        verticalLayout.addItem(spacerItemV)

        # Add all the connections here
        self.defaultsBut.clicked.connect(self.setDefaults)
        self.dicTypeBox.currentIndexChanged.connect(self.changedSettings)
        self.shapeFuncBox.currentIndexChanged.connect(self.changedSettings)
        self.subsetSizeIn.editingFinished.connect(self.changedSettings)
        self.stepSizeIn.editingFinished.connect(self.changedSettings)
        self.refBox.currentIndexChanged.connect(self.changedSettings)
        self.algoTypeBox.currentIndexChanged.connect(self.changedSettings)
        self.startingPIn.editingFinished.connect(self.changedSettings)
        self.convergenceIn.editingFinished.connect(self.changedSettings)
        self.znccTolIn.editingFinished.connect(self.changedSettings)
        self.interpOrderIn.editingFinished.connect(self.changedSettings)
        self.maxIterIn.editingFinished.connect(self.changedSettings)

    # ------------------------------------------------------------------------------
    # Function to get the data from the settings UI and set it in the settings object
    def getData(self, settings):
        settings.DICType = self.dicTypeBox.currentText()
        settings.ShapeFunctions = self.shapeFuncBox.currentText()
        settings.SubsetSize = int(self.subsetSizeIn.text())
        settings.StepSize = int(self.stepSizeIn.text())
        settings.ReferenceStrategy = self.refBox.currentText()
        settings.OptimizationAlgorithm = self.algoTypeBox.currentText()
        settings.StartingPoints = int(self.startingPIn.text())
        settings.ConvergenceThreshold = float(self.convergenceIn.text())
        settings.NZCCThreshold = float(self.znccTolIn.text())
        settings.InterpolationOrder = int(self.interpOrderIn.text())
        settings.MaxIterations = int(self.maxIterIn.text())

    # ------------------------------------------------------------------------------
    # Function to get the data from this class and store it in the settings object
    def setData(self, settings):
        self.dicTypeBox.setCurrentIndex(
            self.dicTypeBox.findText(settings.DICType))
        self.shapeFuncBox.setCurrentIndex(
            self.shapeFuncBox.findText(settings.ShapeFunctions))
        self.subsetSizeIn.setText(str(settings.SubsetSize))
        self.stepSizeIn.setText(str(settings.StepSize))
        self.refBox.setCurrentIndex(
            self.refBox.findText(settings.ReferenceStrategy))
        self.algoTypeBox.setCurrentIndex(
            self.algoTypeBox.findText(settings.OptimizationAlgorithm))
        self.startingPIn.setText(str(settings.StartingPoints))
        self.convergenceIn.setText(str(settings.ConvergenceThreshold))
        self.znccTolIn.setText(str(settings.NZCCThreshold))
        self.interpOrderIn.setText(str(settings.InterpolationOrder))
        self.maxIterIn.setText(str(settings.MaxIterations))

    # ------------------------------------------------------------------------------
    # Function that is called to indicate that the data was changed by the user
    def changedSettings(self):
        self.getData(self.parent.settings)
        self.parent.savedFlag = False
        self.parent.updateWindowTitle()

    # ------------------------------------------------------------------------------
    # Function that is called to indicate that the data was changed by the user
    def setDefaults(self):
        defSettings = sdset.Settings()
        self.setData(defSettings)
        self.changedSettings()

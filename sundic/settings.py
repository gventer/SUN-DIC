import os
import configparser
import sundic.util.datafile as dataFile


class Settings:
    """
    Class that contains the settings for a DIC job.

    The settings are stored in a dictionary that is accessible as an object.
    """

    # Define the default class variables
    __defDebugLevel = 0
    __defImageFolder = 'images'
    __defCPUCount = 1
    __defDICType = 'Planar'
    __defSubsetSize = 33
    __defStepSize = 5
    __defShapeFunctions = 'Affine'
    __defReferenceStrategy = 'Relative'
    __defStartingPoints = 4
    __defGaussianBlurSize = 5
    __defGaussianBlurStdDev = 0.0
    __defDatumImage = 0
    __defTargetImage = -1
    __defIncrement = 1
    __defROI = [0, 0, 0, 0]
    __defBackgroundCutoff = 25
    __defOptimizationAlgorithm = 'IC-GN'
    __defMaxIterations = 50
    __defConvergenceThreshold = 0.0001
    __defNZCCThreshold = 0.999

    # --------------------------------------------------------------------------------------------

    def __init__(self):
        """
        Initialize the DIC settings object to the default values.
        """
        self.DebugLevel = self.__defDebugLevel
        self.ImageFolder = self.__defImageFolder
        self.CPUCount = self.__defCPUCount
        self.DICType = self.__defDICType
        self.SubsetSize = self.__defSubsetSize
        self.StepSize = self.__defStepSize
        self.ShapeFunctions = self.__defShapeFunctions
        self.ReferenceStrategy = self.__defReferenceStrategy
        self.StartingPoints = self.__defStartingPoints
        self.GaussianBlurSize = self.__defGaussianBlurSize
        self.GaussianBlurStdDev = self.__defGaussianBlurStdDev
        self.DatumImage = self.__defDatumImage
        self.TargetImage = self.__defTargetImage
        self.Increment = self.__defIncrement
        self.ROI = self.__defROI
        self.BackgroundCutoff = self.__defBackgroundCutoff
        self.OptimizationAlgorithm = self.__defOptimizationAlgorithm
        self.MaxIterations = self.__defMaxIterations
        self.ConvergenceThreshold = self.__defConvergenceThreshold
        self.NZCCThreshold = self.__defNZCCThreshold

    # --------------------------------------------------------------------------------------------

    @classmethod
    def fromSettingsFile(cls, filename='settings.ini'):
        """
        Load the settings object from a configuration file.

        Args:
            filename (str, optional): The name of the file to load the settings from.

        Returns:
            Settings: The settings object loaded from the file.
        """
        obj = cls.__new__(cls)  # Does not call __init__
        super(Settings, obj).__init__()
        obj.loadSettings(filename)
        return obj

    # --------------------------------------------------------------------------------------------

    @classmethod
    def fromMsgPackFile(cls, filename):
        """
        Load the settings object from a file in MsgPack format.

        Args:
            filename (str): The name of the file to load the MsgPack 
                data from.

        Returns:
            Settings: The settings object loaded from the file.
        """
        obj = cls.__new__(cls)  # Does not call __init__
        super(Settings, obj).__init__()
        obj._loadMsgPackFile_(filename)
        return obj

    # --------------------------------------------------------------------------------------------
    @classmethod
    def fromMsgPackDict(cls, setDict):
        """
        Load the settings object from a dictionary read from the MsgPack format.

        Args:
            setDict (dict): The MsgPack dictionary that was read in.

        Returns:
            Settings: The settings object loaded from the file.
        """
        obj = cls.__new__(cls)  # Does not call __init__
        super(Settings, obj).__init__()
        obj._loadMsgPackDict_(setDict)
        return obj

    # --------------------------------------------------------------------------------------------
    def __repr__(self):
        """
        String representation of the object data values.
        """
        retStr = '\nDIC Job Settings:\n'
        retStr += '------------------------------------------------\n'
        retStr += "  %25s : %s\n" % ('Debug Level', str(self.DebugLevel))
        retStr += "  %25s : %s\n" % ('Image Folder', str(self.ImageFolder))
        retStr += "  %25s : %s\n" % ('CPU Count', str(self.CPUCount))
        retStr += "  %25s : %s\n" % ('DIC Type', str(self.DICType))
        retStr += "  %25s : %s\n" % ('Subset Size', str(self.SubsetSize))
        retStr += "  %25s : %s\n" % ('Step Size', str(self.StepSize))
        retStr += "  %25s : %s\n" % ('Shape Functions',
                                     str(self.ShapeFunctions))
        retStr += "  %25s : %s\n" % ('Reference Strategy',
                                     str(self.ReferenceStrategy))
        retStr += "  %25s : %s\n" % ('Starting Points',
                                     str(self.StartingPoints))
        retStr += "  %25s : %s\n" % ('Gaussian Blur Size',
                                     str(self.GaussianBlurSize))
        retStr += "  %25s : %s\n" % \
            ('Gaussian Blur StdDev', str(self.GaussianBlurStdDev))
        retStr += "  %25s : %s\n" % ('Datum Image', str(self.DatumImage))
        retStr += "  %25s : %s\n" % ('Target Image', str(self.TargetImage))
        retStr += "  %25s : %s\n" % ('Increment', str(self.Increment))
        retStr += "  %25s : %s\n" % ('ROI', str(self.ROI))
        retStr += "  %25s : %s\n" % ('Background Cutoff',
                                     str(self.BackgroundCutoff))
        retStr += "  %25s : %s\n" % \
            ('Optimization Algorithm', str(self.OptimizationAlgorithm))
        retStr += "  %25s : %s\n" % ('Max Iterations', str(self.MaxIterations))
        retStr += "  %25s : %s\n" % \
            ('Convergence Threshold', str(self.ConvergenceThreshold))
        retStr += "  %25s : %s\n" % \
            ('NZCC Threshold', str(self.NZCCThreshold))

        return retStr

    # --------------------------------------------------------------------------------------------

    def _loadMsgPackFile_(self, filename):
        """
        Load the settings object from a file in MsgPack format.

        Args:
            filename (str): The name of the file to load the MsgPack 
                data from.
        """
        # Read msgpack file back
        df = dataFile.DataFile.openReader(filename)
        _, _, setDict = df.readHeading()

        # Set object dictionary from setting read from file
        self._loadMsgPackDict_(setDict)

        # Close the file
        df.close

    # --------------------------------------------------------------------------------------------
    def _loadMsgPackDict_(self, setDict):
        """
        Load the settings object from a MsgPack file dictionary.

        Args:
            setDict (dict): The settings dictionary read from teh MsgPack file.
        """

        # Set object dictionary from setting read from file
        self.__dict__ = setDict

    # --------------------------------------------------------------------------------------------

    def isRelativeStrategy(self):
        """
        Determine if the reference strategy is relative.

        Returns:
            - bool: True if the reference strategy is relative, False otherwise.
        """
        if self.ReferenceStrategy == 'Relative':
            return True
        else:
            return False

    # --------------------------------------------------------------------------------------------

    def isAbsoluteStrategy(self):
        """
        Determine if the reference strategy is absolute.

        Returns:
            - bool: True if the reference strategy is absolute, False otherwise.
        """
        if self.ReferenceStrategy == 'Absolute':
            return True
        else:
            return False

    # --------------------------------------------------------------------------------------------

    def isICGN(self):
        """
        Determine if the optimization algorithm is the Gauss-Newton method.

        Returns:
            - bool: True if the optimization algorithm is the Gauss-Newton method, False otherwise.
        """
        if self.OptimizationAlgorithm == 'IC-GN':
            return True
        else:
            return False

    # --------------------------------------------------------------------------------------------

    def isICLM(self):
        """
        Determine if the optimization algorithm is the Levenberg-Marquardt method.

        Returns:
            - bool: True if the optimization algorithm is the Levenberg-Marquardt method, False otherwise.
        """
        if self.OptimizationAlgorithm == 'IC-LM':
            return True
        else:
            return False

    # --------------------------------------------------------------------------------------------

    def isFastICLM(self):
        """
        Determine if the optimization algorithm is the fast version of the Levenberg-Marquardt method.

        Returns:
            - bool: True if the optimization algorithm is the fast Levenberg-Marquardt method, False otherwise.
        """
        if self.OptimizationAlgorithm == 'Fast-IC-LM':
            return True
        else:
            return False

    # --------------------------------------------------------------------------------------------

    def loadSettings(self, configFile='settings.ini'):
        """
        Load the DIC settings from a configuration file and return them as a dictionary.
        The settings file name is obtained from the constant CONFIG_FILENAME.

        Returns:
            - settings(dict): A dictionary containing the DIC settings.

        Raises:
            - ValueError: If the settings file contains invalid values.
        """
        # Load the configuration file containing the DIC settings
        cp = configparser.ConfigParser(converters={'intlist': lambda x: [
            int(i.strip()) for i in x.split(',')] if len(x) > 0 else []})
        cp.read(configFile)

        # -- General -----------------------------------------------------------------------------
        self.DebugLevel = cp.getint(
            'General', 'DebugLevel', fallback=self.__defDebugLevel)
        if self.DebugLevel < 0:
            self.DebugLevel = 0
            print('WARNING: Config Parser - DebugLevel set to minimum value of 0')
        if self.DebugLevel > 2:
            self.DebugLevel = 2
            print('WARNING: Config Parser - DebugLevel set to maximum value of 2')

        self.ImageFolder = cp.get(
            'General', 'ImageFolder', fallback=self.__defImageFolder)
        if not os.path.isdir(self.ImageFolder):
            raise ValueError('Config Parser:  Image folder does not exist')

        self.CPUCount = cp.get('General', 'CPUCount',
                               fallback=self.__defCPUCount).lower()
        if self.CPUCount == 'auto':
            self.CPUCount = os.cpu_count()
        else:
            self.CPUCount = int(self.CPUCount)

        if self.CPUCount < 1:
            raise ValueError('Config Parser:  CPUCount must be greater than 0')

        self.DICType = cp.get('General', 'DICType', fallback=self.__defDICType)
        if self.DICType not in ['Planar', 'Stereo']:
            raise ValueError(
                'Config Parser:  DICType must be Planar or Stereo')

        # -- DICSettings -------------------------------------------------------------------------
        self.SubsetSize = cp.getint(
            'DICSettings', 'SubSetSize', fallback=self.__defSubsetSize)
        if self.SubsetSize < 1:
            raise ValueError(
                'Config Parser:  Subset size must be greater than 0')
        if self.SubsetSize % 2 == 0:
            raise ValueError(
                'Config Parser:  Subset size must be an odd number')

        self.StepSize = cp.getint(
            'DICSettings', 'StepSize', fallback=self.__defStepSize)
        if self.StepSize < 1:
            raise ValueError(
                'Config Parser:  Step size must be greater than 0')

        self.ShapeFunctions = cp.get(
            'DICSettings', 'ShapeFunctions', fallback=self.__defShapeFunctions)
        if self.ShapeFunctions not in ['Affine', 'Quadratic']:
            raise ValueError(
                'Config Parser:  ShapeFunctions must be Affine or Quadratic')

        self.ReferenceStrategy = cp.get(
            'DICSettings', 'ReferenceStrategy', fallback=self.__defReferenceStrategy)
        if self.ReferenceStrategy not in ['Relative', 'Absolute']:
            raise ValueError(
                'Config Parser:  ReferenceStrategy must be Relative or Absolute')

        self.StartingPoints = cp.getint(
            'DICSettings', 'StartingPoints', fallback=self.__defStartingPoints)
        if self.StartingPoints < 1:
            raise ValueError(
                'Config Parser:  StartingPoints be greater than 0')

        # -- PreProcess --------------------------------------------------------------------------
        self.GaussianBlurSize = cp.getint(
            'PreProcess', 'GaussianBlurSize', fallback=self.__defGaussianBlurSize)
        if self.GaussianBlurSize < 0:
            raise ValueError(
                'Config Parser:  GaussianBlurSize must be greater than or equal to 0')
        if self.GaussianBlurSize % 2 == 0 and self.GaussianBlurSize > 0:
            raise ValueError(
                'Config Parser:  GaussianBlurSize must be an odd number')

        self.GaussianBlurStdDev = cp.getfloat(
            'PreProcess', 'GaussianBlurStdDev', fallback=self.__defGaussianBlurStdDev)
        if self.GaussianBlurStdDev < 0:
            raise ValueError(
                'Config Parser:  GaussianBlurStdDev must be greater than or equal to 0.0')

        # -- ImageSetDefinition ------------------------------------------------------------------
        self.DatumImage = cp.getint(
            'ImageSetDefinition', 'DatumImage', fallback=self.__defDatumImage)
        if self.DatumImage < 0:
            raise ValueError(
                'Config Parser:  DatumImage must be greater than or equal to 0')

        self.TargetImage = cp.getint(
            'ImageSetDefinition', 'TargetImage', fallback=self.__defTargetImage)
        if (self.TargetImage != -1) and (self.TargetImage < self.DatumImage):
            raise ValueError(
                'Config Parser:  TargetImage must be greater than DatumImage')

        self.Increment = cp.getint(
            'ImageSetDefinition', 'Increment', fallback=self.__defIncrement)
        if self.Increment < 1:
            raise ValueError(
                'Config Parser:  Increment must be greater than 0')

        self.ROI = cp.getintlist('ImageSetDefinition',
                                 'ROI', fallback=self.__defROI)
        if min(self.ROI) < 0:
            raise ValueError(
                'Config Parser:  ROI values must be greater than or equal to 0')

        self.BackgroundCutoff = cp.getint(
            'ImageSetDefinition', 'BackgroundCutoff', fallback=self.__defBackgroundCutoff)
        if self.BackgroundCutoff < 0:
            raise ValueError(
                'Config Parser:  BackgroundCutoff value must be greater than or equal to 0')

        # -- Optimization ------------------------------------------------------------------------
        # Initialization of optimisation routine
        self.OptimizationAlgorithm = cp.get(
            'Optimisation', 'OptimizationAlgorithm', fallback=self.__defOptimizationAlgorithm)
        if self.OptimizationAlgorithm not in ['IC-GN', 'IC-LM', 'Fast-IC-LM']:
            raise ValueError(
                'Config Parser:  OptimizationAlgorithm must be IC-GN , IC-LM or Fast-IC-LM')

        self.MaxIterations = cp.getint(
            'Optimisation', 'MaxIterations', fallback=self.__defMaxIterations)
        if self.MaxIterations < 1:
            raise ValueError(
                'Config Parser:  MaxIterations must be greater than 0')

        self.ConvergenceThreshold = cp.getfloat(
            'Optimisation', 'ConvergenceThreshold', fallback=self.__defConvergenceThreshold)
        if self.ConvergenceThreshold < 0:
            raise ValueError(
                'Config Parser:  ConvergenceThreshold must be greater than or equal to 0')

        self.NZCCThreshold = cp.getfloat(
            'Optimisation', 'NZCCThreshold', fallback=self.__defNZCCThreshold)
        if self.NZCCThreshold < 0:
            raise ValueError(
                'Config Parser:  NZCCThreshold must be greater than or equal to 0')

        # Perform Debug output if requested - print all values in dictionary
        if self.DebugLevel > 1:
            print(self.__repr__())

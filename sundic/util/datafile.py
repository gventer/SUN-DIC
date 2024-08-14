import numpy as np
from datetime import datetime
import sundic.version as version
import msgpack as msgpack
import msgpack_numpy as msgp_np

# Setup the msgpack_numpy environment
msgp_np.patch()

class DataFile:
    """
    Class to handle the reading and writing of data files

    The data file is a binary file that contains the following:
        - Version number of the program that created the file
        - Date and time the file was created
        - The settings dictionary
        - The subset displacement data

    The data file is written using the msgpack library to ensure that the
    data is stored in a compact binary format that can be read on any
    platform.

    The data file is read using the msgpack library to ensure that the
    data is read correctly and efficiently.
    """

    __fh__ = None   # The filehandle to use with this object

    # --------------------------------------------------------------------------
    @classmethod
    def openReader(cls, filename):
        """
        Open the data file for reading

        args:
            filename (str) The name of the file to open
        """
        obj = cls.__new__(cls)  # Does not call __init__
        super(DataFile, obj).__init__()
        obj.__fh__ = open(filename, "rb")
        return obj

    # --------------------------------------------------------------------------
    @classmethod
    def openWriter(cls, filename):
        """
        Open the data file for writing
        
        args:
            filename (str) The name of the file to open
        """
        obj = cls.__new__(cls)  # Does not call __init__
        super(DataFile, obj).__init__()
        obj.__fh__ = open(filename, "wb")
        return obj
    

    # --------------------------------------------------------------------------
    def __delattr__(self):
        """
        Close the data file when the object is deleted
        """
        self.__fh__.close()


    # --------------------------------------------------------------------------
    @classmethod
    def close(cls):
        """
        Close the data file
        """
        if cls.__fh__ is not None:
            cls.__fh__.close()
            cls.__fh__ = None

    # --------------------------------------------------------------------------
    def writeHeading(self, settings):
        """
        Write a heading to the data file

        args:
            settings (dict) The settings dictionary to write to the file
        """
        # Write the version number
        pVersion = msgpack.packb( version.__version__ )
        self.__fh__.write(pVersion)

        # Write the date and time
        now = datetime.now()
        pDate = msgpack.packb(now.strftime("%d/%m/%Y %H:%M:%S"))
        self.__fh__.write(pDate)

        # Write the settings dictionary
        pSettings = msgpack.packb(settings.__dict__)
        self.__fh__.write(pSettings)


    # --------------------------------------------------------------------------
    def writeSubSetData(self, imgPair, data):
        """
        Write the subset data to the data file

        args:
            imgPair (int) The image pair ID
            data (numpy.ndarray) The data to write
        """
        # Write the image pair ID
        pImgPair = msgpack.packb(imgPair)
        self.__fh__.write(pImgPair)

        # Write the dimensions of the data
        pDim = msgpack.packb(data.shape)
        self.__fh__.write(pDim)

        # Write the data
        #--------------------------------------------------------------
        # NB all data output from the dic analysis is dumped - this
        # contains the x and y displacment data as well as all the 
        # other shape function data.  The shape function data is not
        # used in the post processing so it is not necessary to write
        # it to the file.  For now the data is preserved in case there
        # is a need for it in the future.  However, an efficiency gain
        # could be made by only writing the x and y displacement data
        # --------------------------------------------------------------
        pData = msgpack.packb(np.ravel(data))
        self.__fh__.write(pData)


    # --------------------------------------------------------------------------
    def readHeading(self):
        """
        Read the heading from the data file

        returns:
            version (str) The version number of the data file
            date (str) The date and time the file was created
            settings (dict) The settings dictionary
        """
        # Go to the start of the file
        self.__fh__.seek(0)

        # Setup the unpacker
        unp = msgpack.Unpacker(self.__fh__, raw=False)
        
        # Get the heading data
        date = unp.unpack()
        version = unp.unpack()
        settings = unp.unpack()

        return version, date, settings

    
    # --------------------------------------------------------------------------
    def readSubSetData(self, imgPair):
        """
        Read the subset data from the data file

        args:
            imgPair (int) The image pair ID

        returns:
            data (numpy.ndarray) The subset data from the file
        """

        # Go to the start of the file - always do this so that we know where we are
        self.__fh__.seek(0)

        # Setup the unpacker and ignore the heading
        unp = msgpack.Unpacker(self.__fh__, raw=False)
        _ = unp.unpack()
        _ = unp.unpack()
        _ = unp.unpack()

        # Loop through the file to find the data
        try :
            while True:
                currImgPair = unp.unpack()
                dim = unp.unpack()
                data = unp.unpack().reshape(dim)
                if currImgPair == imgPair:
                    break
        except msgpack.OutOfData:
            pass

        return data
import numpy as np
from datetime import datetime
import sundic.version as version
import msgpack as msgpack
import msgpack_numpy as msgp_np
import zlib

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

        # Store the save mode and compression flag from settings for later use
        self.dataSaveMode = getattr(settings, 'DataSaveMode', 'All')
        self.dataCompression = getattr(settings, 'DataCompression', True)

        # Write the version number
        pVersion = msgpack.packb(version.__version__)
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

        # Filter data to save only essential columns if 'disp_only' mode is selected
        save_mode = getattr(self, 'dataSaveMode', 'All')
        if save_mode == 'disp_only':
            data_to_write = data[:, :, [0, 1, 2, 3, 4, 5, 11]]
        else:
            data_to_write = data

        # Write the dimensions of the data
        pDim = msgpack.packb(data_to_write.shape)
        self.__fh__.write(pDim)

        # Write the data
        # --------------------------------------------------------------
        # NB all data output from the dic analysis is dumped - this
        # contains the x and y displacment data as well as all the
        # other shape function data.  The shape function data is not
        # used in the post processing so it is not necessary to write
        # it to the file.  For now the data is preserved in case there
        # is a need for it in the future.  However, an efficiency gain
        # could be made by only writing the x and y displacement data
        # --------------------------------------------------------------

        # Pack the raw data
        raw_packed_data = msgpack.packb(np.ravel(data_to_write))
        
        # Compress the data using zlib if enabled
        if getattr(self, 'dataCompression', True):
            compressed_data = zlib.compress(raw_packed_data, level=6)
            pData = msgpack.packb(compressed_data)
        else:
            pData = raw_packed_data
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

        # Skip the header info in the file to get to the subset data
        unp = self._skipHeader_()

        last_data = None
        last_dim = None

        # Loop through the file to find the data
        try:
            while True:
                currImgPair = unp.unpack()
                dim = unp.unpack()
                
                raw_payload = unp.unpack()

                # Check if payload is compressed (bytes) and decompress, 
                # otherwise read normally for backward compatibility with older files
                if isinstance(raw_payload, bytes):
                    decompressed_data = zlib.decompress(raw_payload)
                    data_raw = msgpack.unpackb(decompressed_data).reshape(dim)
                else:
                    data_raw = raw_payload.reshape(dim)
                
                last_data = data_raw
                last_dim = dim
                
                if currImgPair == imgPair:
                    break
        except msgpack.OutOfData:
            pass

        if last_data is None:
            return None

        # If data was saved in 'disp_only' mode (7 columns), pad with zeros 
        # to recreate the 17-column structure expected by the post-processing tools
        if len(last_dim) == 3 and last_dim[2] == 7:
            full_data = np.zeros((last_dim[0], last_dim[1], 17))
            full_data[:, :, [0, 1, 2, 3, 4, 5, 11]] = last_data
            return full_data
        else:
            return last_data
        
    
    
    # --------------------------------------------------------------------------
    def containsResults(self):
        """
        Check if the data file contains any results

        returns:
            bool (bool) True if the file contains results, False otherwise
        """
        # Loop through the file to find the data
        try:

            # Skip the header info in the file to get to the subset data
            unp = self._skipHeader_()

            # Check if there is any image data results           
            while True:
                _ = unp.unpack()  # skip the image pair ID
                dim = unp.unpack()

                # Get the raw payload and check if it's compressed or not
                raw_payload = unp.unpack()

                # Check if payload is compressed (bytes) and decompress, 
                # otherwise read normally for backward compatibility with older files
                if isinstance(raw_payload, bytes):
                    decompressed_data = zlib.decompress(raw_payload)
                    data_raw = msgpack.unpackb(decompressed_data).reshape(dim)
                else:
                    data_raw = raw_payload.reshape(dim) 

                return True

        except msgpack.OutOfData:
            pass

        return False

    # --------------------------------------------------------------------------
    def getNumImagePairs(self):
        """
        Get the number of image pairs in the data file

        returns:
            numImgPairs (int) The number of image pairs in the file
        """

        # Loop through the file to find the data
        numImgPairs = 0
        try:

            # Skip the header info in the file to get to the subset data
            unp = self._skipHeader_()

            # Count the image pairs
            while True:
                _ = unp.unpack() # Skip the image pair ID
                dim = unp.unpack()

                # Get the raw payload and check if it's compressed or not
                raw_payload = unp.unpack()

                # Check if payload is compressed (bytes) and decompress, 
                # otherwise read normally for backward compatibility with older files
                if isinstance(raw_payload, bytes):
                    decompressed_data = zlib.decompress(raw_payload)
                    data_raw = msgpack.unpackb(decompressed_data).reshape(dim)
                else:
                    data_raw = raw_payload.reshape(dim) 

                numImgPairs = numImgPairs + 1

        # Handle any exceptions - like end of the file
        except msgpack.OutOfData:
            pass

        # Return the number of image pairs found
        return numImgPairs


    def _skipHeader_(self):
        """
        Skip the header of the data file

        This is a helper function to skip the header of the data file when reading
        the subset data.  It is used to avoid reading the header multiple times
        when reading multiple image pairs.

        args:
            self (DataFile) The DataFile object
        """
        # Go to the start of the file - always do this so that we know where we are
        self.__fh__.seek(0)

        # Setup the unpacker and ignore the heading
        unp = msgpack.Unpacker(self.__fh__, raw=False, max_buffer_size=0)
        _ = unp.unpack()
        _ = unp.unpack()
        _ = unp.unpack()

        return unp
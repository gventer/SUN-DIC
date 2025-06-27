################################################################################
# This file provides a simple utility that converts an image contained in hdf5 
# container to a tif file format.
#
# Usage: First edit the DIR, EXT and DATA_SET variables to match your needs.
#        Then run the script in the directory containing the hdf5 files.
#        The script will process all files with the specified extension in the
#        current directory.
#        The script will save the converted tif files in the same directory.
#        The script will print the names of the files being processed and any
#        errors encountered during processing.
#        The script will also print the names of all datasets found in the hdf5
#        files.
#        The script currently extracts only one dataset (or image) from the hdf5
#        file. The dataset name is specified in the DATA_SET variable.
#
# Author: G Venter
# Date: 2025/05/12
################################################################################
import h5py
import cv2
import numpy as np
import os

# Define some constants for the script to match your needs
DIR='.'
EXT='.hdf5'
DATA_SET='correlation_load_series_camera_1/camera_pos_1'

# Loop through all files in the current directory and process files
# with the HDF5 extension
for f in os.listdir(DIR):
    if f.endswith(EXT):

        hdf5_file_path = os.path.join(DIR,f)

        # Perform actions on the file here
        print(f"\n-----------------------------------------------------")
        print(f"Processing file: {hdf5_file_path}")
        print(f"-----------------------------------------------------")

        # Example: Open and read the file
        try:
            with h5py.File(hdf5_file_path, "r") as hdf5_file:

                # List all data sets in the file
                print(f"\nList of all data sets in file:")
                print(f"------------------------------")

                def list_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"    Dataset found: {name}")

                hdf5_file.visititems(list_datasets)

                # Now process the specified data set
                if DATA_SET in hdf5_file:

                    print(f"\nProcessing dataset: {DATA_SET}")
                    print(f"------------------------------")

                    # Read the image data from the specified dataset
                    image_data = hdf5_file[DATA_SET][:]

                    # Convert the image to a format that OpenCV can display
                    image = np.array(image_data, dtype=np.uint8)

                    # Display the image using OpenCV
                    print(f"    Saving {hdf5_file_path} to {hdf5_file_path}.tif")
                    cv2.imwrite(f"{hdf5_file_path}.tif", image)
                else:
                    print(f"WARNING:  Dataset '{DATA_SET}' not found in the HDF5 file.")

        except Exception as e:
            print(f"ERROR: Processing {hdf5_file_path} failed: {e}")

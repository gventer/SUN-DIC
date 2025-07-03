__version__ = "0.0.18"
# 0.0.18 - Fixed bug to make GUI work with new zncc convergence check
# 0.0.17 - Added better support of 12 and 16 bit images - these images are now no longer
#          internally converted to 8 bit as was done in the past
#        - Fixed labels in line cut graphs when speciying y-coordinate cuts
#        - Fixed convergence check
#        - Changed internal data structure to allow for subset base subsetsize and
#          subset specific shape functions to be specified
#        - Added support for an optional CZNCC convergence check
# 0.0.16 - Just getting github and pypi in sync
# 0.0.15 - Still working on Ray - now allow connection to an existing Ray cluster
# 0.0.14 - Safe methods for using Ray - retry if necessary
# 0.0.13 - Restructure example packaging for ease of installation
# 0.0.12 - Fixing versions and tags...
# 0.0.11 - Fixing versions and tags...
# 0.0.10 - Fixing versions and tags...
# 0.0.9 - Added ZNCC and ZNSSD data to post-processing and new Fast-IC-LM algorithm to GUI
# 0.0.8 - Improved reading of images and added a new Fast-IC-LM algorithm
# 0.0.7 - Trying to fix run environment for GUI on windows
# 0.0.6 - Publishing to PyPi
# 0.0.5 - Minor change to gradient calculations
# 0.0.4 - Created an settings object and am now serializing the settings object
#         and the results object from the analysis to a msgpack binary file that
#         is used for post-processing
# 0.0.3 - Added support for multiple input files
# 0.0.2 - Changed underlying data structure and added parallel processing
# 0.0.1 - Initial release

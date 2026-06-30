__version__ = "0.0.31"
# 0.0.31 - Improved interpolator for better performance
#        - Minor changes to the readme file and user manual
# 0.0.30 - Fix a problem with _setupSubSets_ when using different subsets sizes for
#          each of the subsets (thus providing a 2D array with subset sizes, rather
#          just a single subset size for all subsets).
#        - Added a user manual
#        - Updated reamdme file and images
# 0.0.29 - Quick bug fix with binary mask setting when opening an existing results file
#        - Fine tune the optional jupyter install
#        - Fine tune the disk size for dilation to a radiius of 1 subset
# 0.0.28 - Allow the user to specify a binary mask to define the region of interest
#        - Make the Jupyter install optional and added to the pyproject.toml file
#        - GUI - added import/export of settings.ini file to File menu
#        - GUI fix - manual changes to the ROI were not saved, but are now
#        - Updated Qt support in results widget
#        - Fixed some issues with the code when the user does not specify a ROI
#          This relates to automatically setting up a usable ROI based on the nominal
#          subset size and automatically resized subsets in cases where the subsets do
#          not all have the same size (e.g. due to being near the edge of the image)
# 0.0.27 - Updated copy_examples
#        - Added dilation support for displacement field to post-processing to help create
#          exclusion zone around automatically detected boundaries in the ROI
#        - Added dilation support to GUI as well 
#        - Update ROI in GUI to work with latest PyQT libraries
# 0.0.26 - Updated requirements file and added pyproject.toml installation 
#          support
# 0.0.25 - Fixed bug in post-processing that caused an exception when
#          not enough matched subsets were found for smoothing
#        - Added cache for to reduce number of function calls and speedup of the code
#        - Cache added for CZNSSD values and relative coordinates in the optimization loop
# 0.0.24 - Fixed some layout issue with the GUI on Windows
#        - Decided to go with the fusion look and feel for the GUI on all platforms
#          to have a consistent look and feel
# 0.0.23 - And we have some icons!!!
#        - Added icons to the GUI
#        - Minor improvements to the newly reworked GUI
#        - Updated the readme file and images
# 0.0.22 - Major refactor of the GUI code with future extensibility and maintainability
#          as focus.  Little change in functionality
#        - Upgraded GUI to PyQt6
#        - Add functionality to cleanly stop the runPlanarDIC thread from the GUI
#        - Added correlation contour plot output to the GUI
#        - Updated the requirements.txt file
# 0.0.21 - Major overhaul with efficiency improvements as focus.  Functionality should not
#          have changed, but performance should be better.
#        - Set default interpolation order to 5th order - not much slower but more accurate
#        - Updated GUI to contain interpolation order selection
# 0.0.20 - Added support for reading results files that are larger than 100MB in size
#        - Fixed issue with upper bounds of automatic ROI definition
#        - Fixed issue with using Quadratic shape functions with IC-LM algorithm not working
#        - Fixed issue with scaling of gradients from Savitzky-Golay filter not accouting
#          for the stepssize of the subsets
#        - Added user setting to control the interpolation order (3rd order of 5th order)
# 0.0.19 - Fixed issue with pure black and white (binary) images during the
#          Akaze detection
#        - Fix subset size that may result in out of bounds subsets relative to the
#          original image size
#        - Fixed issue with relative strategy x and y coordinate updates
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

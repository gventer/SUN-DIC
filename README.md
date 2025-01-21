# SUN-DIC

Stellenbosch University DIC Code

# Important Notice

You are accessing a very early release of the Stellenbosch University DIC Code called SUN-DIC. Currently this code has the following key features and limitations. Please forward any problems and/or suggestions for improvement to the author. More detailed information about the implementation will be provided at a later stage.

## Limitations

1. Can currently only deal with 2D planar problems (a stereo version is planned)
2. Can only specify a rectangular region of interest (ROI), but an all black background in the image pair can be used to automatically deal with irregularly shaped domains.  All black (according to a user specified threshold) background subsets are automatically ignored

## Key features

1.  Completely open source using standard Python libraries wherever possible
2.  Both an easy to use use GUI as well as an API are provided for using the code
3.  Make use of the zero-mean normalized sum of squared differences (ZNSSD) correlation criterion
4.  Have an advanced starting strategy for automatically creating initial guesses at a small number of starting points using the Akaze feature detection algorithm
5.  Have both linear (affine) and quadratic shape functions available
6.  Have both an inverse compositional Gauss-Newton (IC-GN) and an inverse compositional Levenberg-Marquardt (IC-LM) solver implemented
7.  Have both an absolute and relative update strategy when considering multiple image pairs
8.  Calculate both displacements and strains
9.  Strains are calculated using a Savitzky-Golay smoothing operations
10. Parallel computing

# Installation

1. Clone the repository
2. Create a virtual environment
3. Activate the virtual environment
4. Install the cloned environment

Below are some pointers how to achieve this using either python/pip or anacoda

## Using `pip`

1. Create a new virtual environment for use with this package

```
python3 -m venv sundic
```

2. Activate the virtual environment

```
source sundic/bin/activate
```

3. Download and install the package

```
git clone https://github.com/gventer/SUN-DIC.git
pip install ./SUN-DIC
```

## Using `anacoda` - from the command line

1. Create a new virtual environment for use with this package

```
conda create -n sundic pip
```

2. Activate the virtual environment

```
conda activate sundic
```

3. Download and install the package

```
git clone https://github.com/gventer/SUN-DIC.git
pip install ./SUN-DIC
```

# Usage

## Starting the GUI

Simply type `sundic_gui` to launch the GUI.  Using the GUI is straight-forward.  Simply follow the workflow outlined on the left-hand side of the GUI.  Hovering the mouse over any entry will provide pop-up help for that entry.

## Using the API

1. Open the `test_sundic.ipynb` Jupyter Notebook for a detailed and complete working example
2. Open this from the main SUN-DIC directory and be sure to use the virtual environment you created above
3. Note that the general work flow is to modify the `setting.ini` file, perform the DIC analysis and finally to post-process the results
4. The working example is provided as a Jupyter Notebook, but the API can of course also be used directly from a normal Python `.py` file

# API Documentation

API Documentation can be found at the following github pages:

[https://gventer.github.io/SUN-DIC](https://gventer.github.io/SUN-DIC/)

# Acknowledgementss

- The SUN-DIC analysis code is based on work done by Ed Brisley as part of his MEng degree at Stellenbosch University. His MEng thesis is available at the Stellenbosch University library (https://scholar.sun.ac.za/items/7a519bf5-e62b-45cb-82f1-11f4969da23a)
- The interpolator used is `fast_interp` written by David Stein and available under the Apache 2.0 license at: https://github.com/dbstein/fast_interp
- The Savitsky-Golay 2D smoothing algorithm is from the scipy cookbook available at: https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html

# License

This project is licensed under the MIT License - see the LICENSE file for details

# Authors

[Gerhard Venter](https://github.com/gventer/)

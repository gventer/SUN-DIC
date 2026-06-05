# SUN-DIC

---

**Stellenbosch University Digital Image Correlation (DIC) Code**

[![PyPI version](https://img.shields.io/pypi/v/SUN-DIC?style=flat-square&color=2C7BE5)](https://pypi.org/project/SUN-DIC/) [![Python version](https://img.shields.io/pypi/pyversions/SUN-DIC?style=flat-square&color=F0B429)](https://pypi.org/project/SUN-DIC/) [![License: MIT](https://img.shields.io/badge/License-MIT-2EA44F.svg?style=flat-square)](LICENSE)

[![PyPI Downloads](https://img.shields.io/pypi/dm/SUN-DIC?style=flat-square&color=0F766E)](https://pypi.org/project/SUN-DIC/) [![GitHub stars](https://img.shields.io/github/stars/gventer/SUN-DIC.svg?style=flat-square&color=D97706)](https://github.com/gventer/SUN-DIC/stargazers)


---

SUN-DIC is an open-source Python package for **2D digital image correlation (DIC)** developed at **Stellenbosch University**. It provides both a **graphical user interface (GUI)** and a **Python API** for displacement and strain analysis from image sets, making it suitable for both interactive use and research workflows.

> **Early release notice**  
> SUN-DIC is currently in an early public release phase. Core functionality is available and documented, but the interface and documentation will continue to evolve. Bug reports, suggestions, and feedback are very welcome.
> 

---

## Quick Start

### Install with `pip`

> **Note** Please see detailed installation instructions for both `pip` and `conda` further down in this `README` file.

```bash
python3.11 -m venv sundic
source sundic/bin/activate
pip install SUN-DIC
copy-examples
sundic
```

---
## Documentation

SUN-DIC documentation is currently provided through the following resources:

- **User manual**: installation, GUI workflow, and API overview 
  [SUN-DIC User Manual (PDF)](sundic/docs/SUN-DIC_Manual.pdf)

- **Example configuration**: a fully documented `settings.ini` file included with the provided example problem

- **Example notebook**: `test_sundic.ipynb`, copied into your current directory with `copy-examples`

- **GUI tooltips**: GUI options include tooltip descriptions

- **API documentation**: online reference documentation  
  [![Documentation](https://img.shields.io/badge/docs-online-7A3E9D?style=flat-square)](https://gventer.github.io/SUN-DIC/)


---
## Publications

1. Venter, Gerhard and Neaves, Melody, [*SUN-DIC: A Python-Based Open-Source Software Tool for Digital Image Correlation*](https://www.sciencedirect.com/science/article/pii/S0965997825001814), Advances in Engineering Software, Volume 211, 2025.

---

## Key Features

- Fully open-source, using standard Python libraries wherever possible
- Provides both a user-friendly GUI and a programmable API
- Implements the **Zero-Mean Normalized Sum of Squared Differences (ZNSSD)** correlation criterion
- Uses an advanced starting strategy based on the **AKAZE** feature detection algorithm for initial guess generation
- Supports both **linear (affine)** and **quadratic** shape functions
- Includes **Inverse Compositional Gauss-Newton (IC-GN)** and **Inverse Compositional Levenberg-Marquardt (IC-LM)** solvers
- Provides both **absolute** and **relative** update strategies for handling multiple image pairs
- Supports rectangular regions of interest (ROI) and custom ROIs defined by a black/white mask
- Automatically ignores subsets with an all-black background, allowing irregularly shaped domains to be handled naturally
- Computes **displacements** and **strains**, with multiple plotting and visualization options
- Uses **Savitzky-Golay smoothing** for strain calculations, with optional displacement smoothing using the same algorithm
- Supports parallel computing for improved performance
- Easy installation via [PyPI](https://pypi.org/project/SUN-DIC/)

---

## Limitations

- Currently supports **2D planar** DIC problems only
- A stereo / 3D version is under development


---

## Installation

Although SUN-DIC can be installed without creating a virtual environment, using one is strongly recommended for easier dependency management.

> **Note** The `ray` library providing the parallel computing functionality is typically not supported for the latest Python releases on Windows. If you run into a `ray` dependency issue during installation, please try an older version of Python.

### General Steps

1. Create a virtual environment
2. Activate the environment
3. Install the package from PyPI
4. Optionally install Jupyter dependencies
5. Copy the example problem into your current directory using `copy-examples`
   > Optionally you can issue the `copy-examples --manual` command to also copy the user manual to your current directory

The example problem includes:

- `test_sundic.ipynb`
- `settings.ini`
- `planar_images/`

These files provide a practical starting point for both the GUI and API workflows.

---

### Using `pip`

1. Create a virtual environment (e.g., `sundic`):

   ```bash
   python3.11 -m venv sundic
   ```

2. Activate the virtual environment:
   
**Linux / macOS**
```bash
source sundic/bin/activate
```

**Windows (Command Prompt)**
```bash
sundic\Scripts\activate
```

3. Install the base package:

```bash
pip install SUN-DIC
```

4. Optional -- install Jupyter notebook support:

```bash
pip install "SUN-DIC[jupyter]"
```

5. Copy the example problem:

```bash
copy-examples
```

---

### Using `conda`

1. Create a virtual environment with Python 3.11:

```bash
conda create -n sundic python=3.11
```

2. Activate the environment:

```bash
conda activate sundic
```

3. Install the base package:

```bash
pip install SUN-DIC
```

4. Optional -- install Jupyter notebook support:

```bash
pip install "SUN-DIC[jupyter]"
```

5. Copy the example problem:

```bash
copy-examples
```

---

### Installing Directly from GitHub (Advanced Users)

1. Create and activate a virtual environment using either `pip` or `conda` as described above
   
2. Clone the repository and install the package:

```bash
git clone https://github.com/gventer/SUN-DIC.git
pip install ./SUN-DIC
```

3. Optional: install Jupyter notebook support:

```bash
pip install "./SUN-DIC[jupyter]"
```

4. The example problem is available in:

```text
SUN-DIC/sundic/examples
```

---

## Usage

Make sure the virtual environment where `SUN-DIC` is installed is active before proceeding.

### Running the GUI

1. Launch the GUI from a terminal:

```bash
sundic
```

2. Use the `copy-examples` command to copy a complete example to your current directory
   
3. In the GUI, use **File → Import Settings File** to import the example `settings.ini`
   
4. Run the example problem from the **Analysis** panel

5. Perform post-processing using the **Results** panel
   
6. Follow the workflow shown on the left-hand side of the GUI

GUI entries include tooltips describing the available options.

<img src="screenshots/settings.png" width="450"> <img src="screenshots/image_set.png" width="450"> <img src="screenshots/roi.png" width="450">
<img src="screenshots/analyze.png" width="450"> <img src="screenshots/results.png" width="450">


---

### Using the API

1. Use the `copy-examples` command to copy a complete example to your current directory
2. Open `test_sundic.ipynb` for a fully worked example
3. If needed, install the optional Jupyter dependencies:

```bash
pip install "SUN-DIC[jupyter]"
```

A typical API workflow involves:

- modifying the `settings.ini` file
- running the DIC analysis
- post-processing the results

Although the provided example uses a Jupyter notebook, the API can also be used in standard Python scripts.

---

## Support and Feedback

If you encounter a bug, have a feature suggestion, or would like to provide feedback, please open an issue on the GitHub repository.

---

## Citation

If you use SUN-DIC in academic work, please cite the publication listed above.

---

## Presentations

1. 2025-04-17 -- [MOD Research Group Meeting - Overview of SUN-DIC](presentations/2025_sundic_mod.pdf) 


---

## Acknowledgments

- **SUN-DIC analysis code**: based on work by **Ed Brisley** as part of his MEng degree at Stellenbosch University. His thesis is available through the [Stellenbosch University Library](https://scholar.sun.ac.za/items/7a519bf5-e62b-45cb-82f1-11f4969da23a).
- **Interpolator**: uses `fast_interp` by David Stein, licensed under Apache 2.0. Repository: [fast_interp](https://github.com/dbstein/fast_interp)
- **Smoothing algorithm**: implements the 2D Savitzky-Golay algorithm from the [SciPy Cookbook](https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html)
- **GUI development**: initial development by [Elijah Stockhall](https://github.com/EMStockhall/)
- **Graphical design**: Dr. Melody Neaves

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Author

Developed by [Gerhard Venter](https://github.com/gventer/).

---
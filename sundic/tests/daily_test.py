import os
import numpy as np

import sundic.settings as sdset
import sundic.sundic as sdic
import sundic.post_process as sdpp

# ---------------------------------------------------------------------
# Daily regression test for SUN-DIC
#
# Just make sure the code is still running with the latest dependencies
# and that the results are within a reasonable tolerance of the expected 
# values.
#
# This is called from the an github action workflow, but can also be run 
# locally.
#
# For this to run, the settings.ini file and the planar_images directory
# must be present in the same directory as this test file.  They can be
# copied from the sundic/examples directory.
#
# To run, issue the following command
#     pytest -q daily_test.py -s
# ---------------------------------------------------------------------

def test_daily_sundic_example_regression():

    # The settings (input) and results (output) files
    settings_file = "settings.ini"
    results_file = "results.sdic"

    # Make sure the settings file exists before running the test
    assert os.path.exists(settings_file), f"Missing required file: {settings_file}"

    # Make sure the planar_images directory exists before running the test
    assert os.path.exists("planar_images"), "Missing required directory: planar_images"

    # Load the settings file and set the debug level to 0 (no debug output)
    settings = sdset.Settings.fromSettingsFile(settings_file)
    settings.DebugLevel = 0
    #print(settings.__repr__())

    # Run the planar DIC analysis
    sdic.planarDICLocal(settings, results_file)

    # Get the displacements and check that they are reasonable
    results, nRow, nCols = sdpp.getDisplacements(results_file, -1, smoothWindow=15)
    assert results is not None and results.size > 0
    assert nRow > 0 and nCols > 0

    # Remove the rows with NaN values (if any) and check that the displacements are finite
    results = results[~np.isnan(results).any(axis=1)]
    assert results.size > 0

    # Get the displacement componets and check that they are finite
    ux = results[:, sdpp.DispComp.X_DISP]
    uy = results[:, sdpp.DispComp.Y_DISP]
    um = results[:, sdpp.DispComp.DISP_MAG]
    assert np.isfinite(ux).all()
    assert np.isfinite(uy).all()
    assert np.isfinite(um).all()

    # Calculate stats on the displacements and compare to expected values
    stats = {
        "ux_min": float(np.min(ux)),
        "ux_max": float(np.max(ux)),
        "ux_mean": float(np.mean(ux)),
        "ux_std": float(np.std(ux)),
        "uy_min": float(np.min(uy)),
        "uy_max": float(np.max(uy)),
        "uy_mean": float(np.mean(uy)),
        "uy_std": float(np.std(uy)),
        "um_min": float(np.min(um)),
        "um_max": float(np.max(um)),
        "um_mean": float(np.mean(um)),
        "um_std": float(np.std(um)),
    }
    expected = {
        "ux_min": -41.3179,
        "ux_max": -6.8583,
        "ux_mean": -24.2743,
        "ux_std": 14.9013,
        "uy_min": -8.5182,
        "uy_max": 9.7297,
        "uy_mean": 0.2014,
        "uy_std": 1.9595,
        "um_min": 6.8510,
        "um_max": 41.2756,
        "um_mean": 24.3615,
        "um_std": 14.8894,
    }

    # Tolerance: tune if needed for platform/library drift.
    atol = 0.005

    # Make the comparison on the results
    for k, exp in expected.items():
        val = stats[k]
        assert abs(val - exp) <= atol, f"{k}: got {val:.4f}, expected {exp:.4f}, |Δ|={abs(val-exp):.4f} > {atol}"

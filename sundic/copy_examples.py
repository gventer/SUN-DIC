################################################################################
# This file contains the functions to install example problems by copying the
# examples installed by setup.py from the resources directory of the sundic 
# package to the current working directory.
##
# Author: G Venter
# Date: 2025/04/14
################################################################################
import os
import shutil
from pkg_resources import resource_filename

def copy_examples():

    # List of resources to copy
    resources = [
        ('examples/settings.ini', 'settings.ini'),
        ('examples/test_sundic.ipynb', 'test_sundic.ipynb'),
        ('examples/planar_images', 'planar_images'),
    ]

    # Current working directory as target
    target_dir = os.getcwd()

    # Copy all the resources to the target directory
    for resource, target_name in resources:
        # Get the full path of the resource in the package
        source_path = resource_filename('sundic', resource)

        # Check if the resource is a file or a directory
        if os.path.isfile(source_path):
            shutil.copy(source_path, os.path.join(target_dir, target_name))
            print(f"Copied {resource} to {target_dir}")
        elif os.path.isdir(source_path):
            target_path = os.path.join(target_dir, target_name)
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            print(f"Copied directory {resource} to {target_dir}")
        else:
            print(f"Warning: Resource {resource} not found in package.")

if __name__ == "__main__":
    copy_examples()

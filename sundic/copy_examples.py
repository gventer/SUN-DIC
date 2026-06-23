################################################################################
# This file contains functions to copy example problems installed as package
# resources from the sundic package to the current working directory.
#
# Author: G Venter
# Date: 2025/04/14
################################################################################

from pathlib import Path
import shutil
import argparse
from importlib.resources import files


def _copy_resource(source, target: Path) -> None:
    """
    Copy a package resource (file or directory) to a target path.
    """
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)

    elif source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            _copy_resource(item, target / item.name)

    else:
        raise FileNotFoundError(f"Resource not found: {source}")


def copy_examples(include_manual: bool = False) -> list[Path]:
    """
    Copy example files from the installed sundic package to the current
    working directory.

    Parameters
    ----------
    include_manual : bool, optional
        If True, also copy the user manual PDF into a local 'docs' directory.
        Default is False.

    Returns
    -------
    list[Path]
        A list of top-level paths copied to the target directory.
    """

    resources = [
        ("examples/settings.ini", "settings.ini"),
        ("examples/test_sundic.ipynb", "test_sundic.ipynb"),
        ("examples/planar_images", "planar_images"),
    ]

    if include_manual:
        resources.append(("docs/SUN-DIC_Manual.pdf", "docs/SUN-DIC_Manual.pdf"))

    target_dir = Path.cwd()
    package_root = files("sundic")
    copied_paths = []

    for resource, target_name in resources:
        source = package_root.joinpath(resource)
        target = target_dir / target_name

        try:
            _copy_resource(source, target)
            print(f"Copied {resource} -> {target}")
            copied_paths.append(target)
        except Exception as exc:
            print(f"Error copying {resource}: {exc}")

    return copied_paths


def main() -> None:
    """
    Command-line entry point for copying example files.
    """
    parser = argparse.ArgumentParser(
        description="Copy SUN-DIC example files to the current working directory."
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Also copy the SUN-DIC user manual PDF into a local docs directory.",
    )

    args = parser.parse_args()
    copy_examples(include_manual=args.manual)


if __name__ == "__main__":
    main()
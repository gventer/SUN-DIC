import io
from setuptools import setup

requirements = {}
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

version = {}
with open("sundic/version.py") as f:
    exec(f.read(), version)

setup(
    name='SUN-DIC',
    version=version["__version__"],
    description='Stellenbosch University Digital Image Correlation Library',
    author='Gerhard Venter',
    author_email='gventer@sun.ac.za',
    packages=['sundic', 'sundic.util', 'sundic.gui'],
    include_package_data=True,
    package_data={
        "sundic": [
            "examples/settings.ini",
            "examples/test_sundic.ipynb",
            "examples/planar_images/*",
        ],
    },
    entry_points={
        "console_scripts": [
            "sundic = sundic.gui.sundic_gui:main",
            "copy-examples = sundic.copy_examples:copy_examples",
        ]
    },
    url='https://github.com/gventer/SUN-DIC',
    license='MIT License',
    long_description=io.open('README.md', encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    platforms=['any'],
    install_requires=requirements,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    python_requires=">3.11",
)

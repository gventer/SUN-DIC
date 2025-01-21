import io
from setuptools import setup

requirements = {}
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

version = {}
with open("sundic/version.py") as f:
    exec(f.read(), version)

setup(
    name='sundic',
    version=version["__version__"],
    description='Stellenbosch University Digital Image Correlation Library',
    author='Gerhard Venter',
    author_email='gventer@sun.ac.za',
    packages=['sundic', 'sundic.util', 'sundic.gui'],
    entry_points={
        "gui_scripts": [
            "sundic = sundic.gui.sundic_gui:main"
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
    python_requires=">3.11"
)

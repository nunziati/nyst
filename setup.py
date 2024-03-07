r"""setup.py"""

import os

from setuptools import setup


def read(fname):
    r"""Reads a file and returns its content as a string."""
    return open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8").read()


def get_version(rel_path):
    r"""Gets the version of the package from the __init__ file."""
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


setup(
    name="nyst",
    version=get_version("nyst/__init__.py"),
    description="Package for face/eye video analysis and nystagmus detection",
    url="https://github.com/nunziati/nyst",
    author="Giacomo Nunziati",
    author_email="giacomo.nunziati.0@gmail.com",
    license="MIT License",
    keywords="eye tracking pupil nystagmus video",
    long_description=read("README.md"),
    # entry_points={
    #     'console_scripts': [
    #         'sadic = sadic.cli_sadic:main',
    #     ],
    # },
    packages=[
        "nyst",
        "nyst.analysis",
        "nyst.pipeline",
        "nyst.pupil",
        "nyst.roi",
        "nyst.utils",
        "nyst.visualization"
    ],
    install_requires=["numpy", "scipy", "matplotlib", "opencv-python"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.12",
    ],
)
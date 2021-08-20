#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="audiotools",
    version="0.02",
    author="Jörg Encke",
    author_email="joerg.encke@posteo.de",
    description="Toolbox for generating and working with audio signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://jencke.github.io/audiotools/",
    python_requires='>=3',
    # packages=find_packages(),
    package_dir={"" : "audiotools"},
    packages=find_packages(where="audiotools"),
    install_requires=[
        'numpy>=1.19',
        'scipy>=1.5.1',
        'matplotlib>=3.3.1',
        'pytest>=5.4.3',
        'coverage'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering"
    ],
)

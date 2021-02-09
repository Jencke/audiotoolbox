#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = "audiotools",
    version = "0.01",
    packages = find_packages(),
    author = "Joerg Encke",
    author_email = "joerg.encke@uol.de",
    description = "Toolbox for generating signals used in auditory research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://jencke.github.io/audiotools/",
    python_requires='>=3',
    install_requires=[
        'numpy>=1.19',
        'scipy>=1.5.1',
        'matplotlib>=3.3.1',
        'pytest>=5.4.3',
    ],
    license = "GPL",
)

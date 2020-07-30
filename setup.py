#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name = "audiotools",
    version = "0.01",
    packages = find_packages(),
    author = "Joerg Encke",
    author_email = "joerg.encke@tum.de",
    description = "Toolbox of some simple functions for audio signal generation",
    python_requires='>=3',
    license = "GPL",
)

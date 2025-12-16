#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import setup, find_packages

setup(
    name='discoveryworld',
    description="DiscoveryWorld - Making new discovery.",
    author='Peter Jansen, Marc-Alexandre Côté',
    version=open(os.path.join("discoveryworld", "version.py")).readlines()[0].split("=")[-1].strip("' \n"),
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/allenai/discoveryworld",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
    ]
)

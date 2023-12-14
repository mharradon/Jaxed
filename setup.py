#!/usr/bin/env python3

import os
from setuptools import setup

# get key package details from __version__.py
about = {}  # type: ignore
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'jaxed', '__version__.py')) as f:
    exec(f.read(), about)

# package configuration - for reference see:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#id9
setup(version=about['__version__'])

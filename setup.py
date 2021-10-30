#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

setup(name = 'atl02v',
      description = 'Verification package for Level-1B ATLAS product.',
      author = 'C Gosmeyer',
      url = 'git@github.com:cgosmeyer/atl02v.git',
      packages = find_packages(),
      install_requires = ['numpy', 'pandas', 'scipy'],
      include_package_data=True
)

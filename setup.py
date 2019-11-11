#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) DeepClean Group (2019)

import pkg_resources
from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    install_requires = [str(r) for r in pkg_resources.parse_requirements(f)]

__version__ = '0.1.0'
    
setup(
    name='Logger',
    version=__version__,
    author='Tri Nguyen',
    author_email='tri.vt.nguyen@gmail.com',
    maintainer='Tri Nguyen',
    maintainer_email='tri.vt.nguyen@gmail.com',
    license='LICENSE.txt',
    description='Deep learning approach to nonlinear regression',
    long_description=open('README.md').read(),
    packages=find_packages(),
    classifiers=(
      'Programming Language :: Python',
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Intended Audience :: End Users/Desktop',
      'Intended Audience :: Developers',
      'Natural Language :: English',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering :: Astronomy',
      'Topic :: Scientific/Engineering :: Physics',
      'Operating System :: POSIX',
      'Operating System :: Unix',
      'Operating System :: MacOS',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ),
    install_requires=install_requires,
    python_requires='>=3.5',
)


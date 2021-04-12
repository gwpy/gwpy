# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2016-2020)
#
# This file is part of GWpy.
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

"""Setup the GWpy package
"""

from setuptools import (
    find_packages,
    setup,
)

# local module to handle build customisations
from setup_utils import (
    CMDCLASS,
    VERSION,
    get_setup_requires,
)

# read description
with open('README.md', 'rb') as f:
    longdesc = f.read().decode().strip()

# -- dependencies -----------

# build dependencies (dynamic based on arguments)
setup_requires = get_setup_requires()

# runtime dependencies
install_requires = [
    'astropy >= 3.0.0',
    'dqsegdb2',
    'gwdatafind',
    'gwosc >= 0.5.3',
    'h5py >= 2.7.0',
    'ligo-segments >= 1.0.0',
    'ligotimegps >= 1.2.1',
    'matplotlib >= 3.3.0',
    'numpy >= 1.15.0',
    'python-dateutil',
    'scipy >= 1.2.0',
    'tqdm >= 4.10.0',
]

# test dependencies
tests_require = [
    "beautifulsoup4",
    "freezegun >= 0.2.3",
    "pytest >= 3.3.0",
    "pytest-cov >= 2.4.0",
]

# -- run setup ----------------------------------------------------------------

setup(
    # metadata
    name='gwpy',
    provides=['gwpy'],
    version=VERSION,
    description="A python package for gravitational-wave astrophysics",
    long_description=longdesc,
    long_description_content_type='text/markdown',
    author='Duncan Macleod',
    author_email='duncan.macleod@ligo.org',
    license='GPL-3.0-or-later',
    url="https://gwpy.github.io",
    download_url="https://gwpy.github.io/docs/stable/install/",
    project_urls={
        "Bug Tracker": "https://github.com/gwpy/gwpy/issues",
        "Discussion Forum": "https://gwpy.slack.com",
        "Documentation": "https://gwpy.github.io/docs/",
        "Source Code": "https://github.com/gwpy/gwpy",
    },

    # package content
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "gwpy-plot=gwpy.cli.gwpy_plot:main",
        ],
    },
    include_package_data=True,

    # dependencies
    cmdclass=CMDCLASS,
    python_requires=">=3.6",
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,

    # classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        ('License :: OSI Approved :: '
         'GNU General Public License v3 or later (GPLv3+)'),
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)

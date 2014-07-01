#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

from __future__ import print_function

import sys
if sys.version < '2.6':
    raise ImportError("Python versions older than 2.6 are not supported.")

import glob
import hashlib
import os.path
import subprocess

try:
    import setuptools
except ImportError:
    import ez_setup
    ez_setup.use_setuptools()
finally:
    from setuptools import (setup, find_packages)
    from setuptools.command import (build_py, egg_info)

from distutils import log
from distutils.dist import Distribution
from distutils.cmd import Command
from distutils.command.clean import (clean, log, remove_tree)

# test for OrderedDict
extra_install_requires = []
try:
    from collections import OrderedDict
except ImportError:
    extra_install_requires.append('ordereddict>=1.1')

# import sphinx commands
try:
    from sphinx.setup_command import BuildDoc
except ImportError:
    cmdclass = {}
else:
    cmdclass = {'build_sphinx': BuildDoc}

# set basic metadata
PACKAGENAME = 'gwpy'
AUTHOR = 'Duncan Macleod'
AUTHOR_EMAIL = 'duncan.macleod@ligo.org'
LICENSE = 'GPLv3'

VERSION_PY = os.path.join(PACKAGENAME, 'version.py')


# -----------------------------------------------------------------------------
# Clean up, including Sphinx, and setup_requires eggs

class GWpyClean(clean):
    def run(self):
        if self.all:
            # remove docs
            sphinx_dir = os.path.join(self.build_base, 'sphinx')
            if os.path.exists(sphinx_dir):
                remove_tree(sphinx_dir, dry_run=self.dry_run)
            else:
                log.warn("%r does not exist -- can't clean it", sphinx_dir)
            # remove version.py
            for vpy in [VERSION_PY, VERSION_PY + 'c']:
                if os.path.exists(vpy) and not self.dry_run:
                    log.info('removing %r' % vpy)
                    os.unlink(vpy)
                elif not os.path.exists(vpy):
                    log.warn("%r does not exist -- can't clean it", vpy)
            # remove setup eggs
            for egg in glob.glob('*.egg'):
                if os.path.isdir(egg):
                    remove_tree(egg, dry_run=self.dry_run)
                else:
                    log.info('removing %r' % egg)
                    os.unlink(egg)
            # remove Portfile
            portfile = 'Portfile'
            if os.path.exists(portfile) and not self.dry_run:
                log.info('removing %r' % portfile)
                os.unlink(portfile)
        clean.run(self)

cmdclass['clean'] = GWpyClean


# -----------------------------------------------------------------------------
# Custom builders to write version.py

class GitVersionMixin(object):
    """Mixin class to add methods to generate version information from git.
    """
    def write_version_py(self, pyfile):
        """Generate target file with versioning information from git VCS
        """
        log.info("generating %s" % pyfile)
        import vcs
        gitstatus = vcs.GitStatus()
        try:
            with open(pyfile, 'w') as fobj:
                gitstatus.write(fobj, author=AUTHOR, email=AUTHOR_EMAIL)
        except:
            if os.path.exists(pyfile):
                os.unlink(pyfile)
            raise
        return gitstatus

    def update_metadata(self):
        """Import package base and update distribution metadata
        """
        import gwpy
        self.distribution.metadata.version = gwpy.__version__
        desc, longdesc = gwpy.__doc__.split('\n', 1)
        self.distribution.metadata.description = desc
        self.distribution.metadata.long_description = longdesc.strip('\n')


class GWpyBuildPy(build_py.build_py, GitVersionMixin):
    """Custom build_py command to deal with version generation
    """
    def __init__(self, *args, **kwargs):
        build_py.build_py.__init__(self, *args, **kwargs)

    def run(self):
        try:
            self.write_version_py(VERSION_PY)
        except ImportError:
            raise
        except:
            if not os.path.isfile(VERSION_PY):
                raise
        self.update_metadata()
        build_py.build_py.run(self)

cmdclass['build_py'] = GWpyBuildPy


class GWpyEggInfo(egg_info.egg_info, GitVersionMixin):
    """Custom egg_info command to deal with version generation
    """
    def finalize_options(self):
        try:
            self.write_version_py(VERSION_PY)
        except ImportError:
            raise
        except:
            if not os.path.isfile(VERSION_PY):
                raise
        if not self.distribution.metadata.version:
            self.update_metadata()
        egg_info.egg_info.finalize_options(self)

cmdclass['egg_info'] = GWpyEggInfo


# -----------------------------------------------------------------------------
# Build Portfile

class BuildPortfile(Command, GitVersionMixin):
    """Generate a Macports Portfile for this project from the current build
    """
    description = 'Generate Macports Portfile'
    user_options = [
       ('version=', None, 'the X.Y.Z package version'),
       ('portfile=', None, 'target output file, default: \'Portfile\''),
       ('template=', None, 'Portfile template, default: \'Portfile.template\''),
    ]

    def initialize_options(self):
        self.version = None
        self.portfile = 'Portfile'
        self.template = 'Portfile.template'
        self._template = None

    def finalize_options(self):
        from jinja2 import Template
        with open(self.template, 'r') as t:
            self._template = Template(t.read())

    def run(self):
        # get version from distribution
        if self.version is None:
            try:
                self.update_metadata()
            except ImportError:
                self.run_command('sdist')
                self.update_metadata()
        # find dist file
        dist = os.path.join(
            'dist',
            '%s-%s.tar.gz' % (self.distribution.get_name(),
                              self.distribution.get_version()))
        # run sdist if needed
        if not os.path.isfile(dist):
            self.run_command('sdist')
            self.update_metadata()
        # get checksum digests
        log.info('reading distribution tarball %r' % dist)
        with open(dist, 'rb') as fobj:
            data = fobj.read()
        log.info('recovered digests:')
        digest = dict()
        digest['rmd160'] = self._get_rmd160(dist)
        for algo in [1, 256]:
            digest['sha%d' % algo] = self._get_sha(data, algo)
        for key, val in digest.iteritems():
            log.info('    %s: %s' % (key, val))
        # write finished portfile to file
        with open(self.portfile, 'w') as fport:
            fport.write(self._template.render(
                version=self.distribution.get_version(), **digest))
        log.info('portfile written to %r' % self.portfile)

    @staticmethod
    def _get_sha(data, algorithm=256):
        hash_ = getattr(hashlib, 'sha%d' % algorithm)
        return hash_(data).hexdigest()

    @staticmethod
    def _get_rmd160(filename):
        p = subprocess.Popen(['openssl', 'rmd160', filename],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise subprocess.CalledProcessError(err)
        else:
            return out.splitlines()[0].rsplit(' ', 1)[-1]

cmdclass['port'] = BuildPortfile


# -----------------------------------------------------------------------------
# Process complicated dependencies

# XXX: this can be removed as soon as a stable release of glue can
#      handle pip/--user
try:
    from glue import git_version
except ImportError as e:
    e.args = ("GWpy requires the GLUE package, which isn\'t available from "
              "PyPI.\nPlease visit\n"
              "https://www.lsc-group.phys.uwm.edu/daswg/projects/glue.html\n"
              "to download and install it manually.",)
    raise

# don't use setup_requires if just checking for information
# (credit: matplotlib/setup.py)
setup_requires = []
if not '--help' in sys.argv:
    dist_ = Distribution({'cmdclass': cmdclass})
    dist_.parse_config_files()
    dist_.parse_command_line()
    if not (any('--' + opt in sys.argv for opt in
            Distribution.display_option_names + ['help']) or
            dist_.commands == ['clean']):
        setup_requires = ['tornado', 'numpy >= 1.5', 'jinja2', 'gitpython']

# -----------------------------------------------------------------------------
# Find files

# Use the find_packages tool to locate all packages and modules
packagenames = find_packages()

# glob for all scripts
if os.path.isdir('bin'):
    scripts = glob.glob(os.path.join('bin', '*'))
else:
    scripts = []

# -----------------------------------------------------------------------------
# run setup

setup(name=PACKAGENAME,
      provides=[PACKAGENAME],
      version=None,
      description=None,
      long_description=None,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url='https://gwpy.github.io/',
      packages=packagenames,
      include_package_data=True,
      cmdclass=cmdclass,
      scripts=scripts,
      setup_requires=setup_requires,
      requires=[
          'glue',
          'dateutil',
          'numpy',
          'scipy',
          'matplotlib',
          'astropy',
          'six',
      ],
      install_requires=[
          'python-dateutil',
          'numpy >= 1.5',
          'scipy',
          'matplotlib >= 1.3.0',
          'astropy >= 0.3',
          'six >= 1.5',
      ] + extra_install_requires,
      extras_require={
          'nds': ['nds2-client'],
          'gwf': ['frameCPP'],
          'doc': ['sphinx'],
      },
      dependency_links=[
          'https://www.lsc-group.phys.uwm.edu/daswg/download/'
              'software/source/glue-1.46.tar.gz#egg=glue-1.46',
      ],
      test_suite='gwpy.tests',
      use_2to3=False,
      classifiers=[
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
      ],
      )

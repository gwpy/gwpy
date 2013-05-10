# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import numpy
from distutils.extension import Extension

from astropy import setup_helpers


def get_extensions():
    include_dirs = []
    libraries = []

    sources = ["cextern/libframe/frgetvect.c"]
    include_dirs = [numpy.get_include(), "/opt/local/include"]

    fr_ext = Extension(
        name="gwpy.io.gwf.__frgetvect",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        language="c",)

    return [fr_ext]

def get_external_libraries():
    return ['libframe']


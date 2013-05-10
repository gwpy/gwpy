# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from distutils.extension import Extension

def get_extensions():
    libraries = []
    sources = ["cextern/segments/infinity.c",
               "cextern/segments/segment.c",
               "cextern/segments/segmentlist.c",
               "cextern/segments/segments.c"]
    include_dirs = ['cextern/segments']

    seg_ext = Extension(
        name="gwpy.segments.__segments",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        language="c",)

    return [seg_ext]

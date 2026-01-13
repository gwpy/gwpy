##############################
Converting GPS times with GWpy
##############################

The command-line tool `gwpy-tconvert` provides a convenient way
to convert to and from GPS times using GWpy.

`gwpy-tconvert` is just a simple wrapper around :meth:`gwpy.time.tconvert`;
for more details, see :doc:`../time/index`.

=====
Usage
=====

.. argparse::
    :ref: gwpy.time.__main__.create_parser
    :prog: gwpy-tconvert

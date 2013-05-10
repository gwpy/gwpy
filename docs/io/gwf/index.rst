*************************
GW frames (`gwpy.io.gwf`)
*************************

============
Introduction
============

`gwpy.io.gwf` provides methods for reading Gravitational Wave Frame files (GWF). These files are written at the observatories, and typicall contain the time-streams for each of the data channels used to record gravitational wave amplitude, and auxiliary signals for characterising the instrument.

===============
Getting started
===============

Reading GWF files
-----------------

Any GWF file can be read with the |read| function as follows::

    >>> from gwpy.io import gwf
    >>> data = gwf.read('data.gwf', channel='G1:DER_DATA_H')

============
Dependencies
============
This module relies on the `core frame library <http://lappweb.in2p3.fr/virgo/FrameL/>`_. The `libframe` package can be installed via `macports <http://www.macports.org>` on Mac OS, or by following `these instructions <http://lappweb.in2p3.fr/virgo/FrameL/FrDoc.html#The_Frame_Library_installation>` for any UNIX-based system.

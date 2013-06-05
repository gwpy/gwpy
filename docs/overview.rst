********
Overview
********

The GWpy package contains classes and utilities providing tools and methods for studying data from gravitational wave detectors, for astrophysical or instrumental purposes.

This package is meant for users who don't care how the code works necessarily, but want to perform some analysis on some data using a tool. As a result this package is meant to be as easy-to-use as possible, coupled with extensive documentation of all functions and standard examples of how to use them sensibly.

The core Python infrastructure is influenced by the `astropy <http://astropy.org>`_ package, a superb set of tools for astrophysical analysis, primarily for FITS images.

This core has been augmented using the `LIGO Algorithm Library Suite (LALSuite) <https://www.lsc-group.phys.uwm.edu/daswg/projects/lalsuite.html>`_, a large collection of ``C99`` routines for analysis and manipulation of data from gravitational wave detectors. These packages use the `SWIG` program to produce python wrappings for all C modules, allowing the GWpy package to leverage both the completeness, and the speed, of these libraries.

In the end, this package has begged, borrowed, and stolen a lot of code from other sources, but should end up packaging them together in a way that makes the whole set easier to use.

============
Why use GWpy
============

A number of users might ask the question:

    I have LAL and PyLAL installed, why should I use GWpy instead?

LAL provides a huge library of `C` functions for analysis of GW detector data. PyLAL provides a layer on top of that for python users to build plots and that sort of thing.

Unfortunately, neither of these packages are written in a 'pythonic' way, so getting things done is a combination of guessing, luck, and trial-and-error (learn by doing).

GWpy hopes to sort that out by provide a properly object-oriented package for easy analysis of GW data, coupled with good documentation.

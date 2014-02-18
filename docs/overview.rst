#############
What is GWpy?
#############

The GWpy package contains classes and utilities providing tools and methods for studying data from gravitational-wave detectors, for astrophysical or instrumental purposes.

This package is meant for users who don't care how the code works necessarily, but want to perform some analysis on some data using a (Python) tool.
As a result this package is meant to be as easy-to-use as possible, coupled with extensive documentation of all functions and standard examples of how to use them sensibly.

The core Python infrastructure is influenced by, and extends the functionaity of, the `astropy <http://astropy.org>`_ package, a superb set of tools for astrophysical analysis, primarily for FITS images.
Additionally, much of the methodology has been derived from, and augmented by, the `LIGO Algorithm Library Suite (LALSuite) <https://www.lsc-group.phys.uwm.edu/daswg/projects/lalsuite.html>`_, a large collection of primarily ``C99`` routines for analysis and manipulation of data from gravitational-wave detectors.
These packages use the `SWIG <http://www.swig.org>` program to produce Python wrappings for all ``C`` modules, allowing the GWpy package to leverage both the completeness, and the speed, of these libraries.

In the end, this package has begged, borrowed, and stolen a lot of code from other sources, but should end up packaging them together in a way that makes the whole set easier to use.

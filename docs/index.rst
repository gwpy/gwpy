################
Welcome to GWpy!
################

GWpy is a collaboration-driven `python <http://www.python.org>`_ package containing a set of common tools for characterising and analysing data from ground-based gravitational-wave detectors and studying the astrophysics associated with gravitational-wave emission.
The package provides a user-friendly, intuitive interface to the common time-domain and frequency-domain data produced by these instruments and their analysis, with rich functionality to help users extract the maximum amount of information from them.

Documentation
=============

First steps
~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   What is GWpy? <overview>
   How do I install GWpy? <install>
   What should I do first? <getting_started>

Working with gravitational-wave interferometer data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, let's consider the core objects used to represent instrumental data:

.. toctree::
   :maxdepth: 2

   detector/channel
   timeseries/index
   spectrum/index
   spectrogram/index
   segments/index

After that, there are a few more advanced objects that extend the functionality of those above:

.. toctree::
   :maxdepth: 2

   timeseries/statevector
   spectrum/variance

Visualisation
~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   plotter/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
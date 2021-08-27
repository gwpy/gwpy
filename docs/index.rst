#########
|_title_|
#########

.. |_title_| image:: gwpy-docs-logo.png

.. title:: GWpy docs

.. ifconfig:: 'dev' in release

   .. warning::

      You are viewing documentation for a development build of GWpy.
      This version may include unstable code, or breaking changes relative
      the most recent stable release.
      To view the documentation for the latest stable release of GWpy, please
      `click here <../stable/>`_.

GWpy is a collaboration-driven `Python <http://www.python.org>`_ package
providing tools for studying data from ground-based gravitational-wave
detectors.

GWpy provides a user-friendly, intuitive interface to the common time-domain
and frequency-domain data produced by the `LIGO <http://www.ligo.org>`_ and
`Virgo <http://www.ego-gw.it>`_ instruments and their analysis,
with easy-to-follow tutorials at each step.

.. image:: https://img.shields.io/conda/vn/conda-forge/gwpy
    :target: https://anaconda.org/conda-forge/gwpy/
    :alt: GWpy Conda-forge version badge
.. image:: https://img.shields.io/pypi/v/gwpy
    :target: https://pypi.org/project/gwpy/
    :alt: GWpy PyPI version badge
.. image:: https://zenodo.org/badge/9979119.svg
    :target: https://zenodo.org/badge/latestdoi/9979119
    :alt: GWpy DOI badge
.. image:: https://img.shields.io/pypi/l/gwpy.svg
    :target: https://choosealicense.com/licenses/gpl-3.0/
    :alt: GWpy license badge

***********
First steps
***********

.. toctree::
   :maxdepth: 1

   What is GWpy? <overview>
   How do I install GWpy? <install>
   Citing GWpy <citing>

*****************
Working with data
*****************

.. toctree::
   :maxdepth: 2
   :caption: Data structures

   timeseries/index
   spectrum/index
   spectrogram/index
   timeseries/statevector
   segments/index
   table/index

.. toctree::
   :maxdepth: 2
   :caption: Data manipulation

   signal/index

.. toctree::
   :maxdepth: 2
   :caption: Visualising data

   plot/index
   cli/index

.. toctree::
   :maxdepth: 1
   :caption: Other utilities

   detector/channel
   time/index
   astro/index
   env

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/timeseries/index
   examples/signal/index
   examples/frequencyseries/index
   examples/spectrogram/index
   examples/segments/index
   examples/table/index
   examples/miscellaneous/index

.. toctree::
   :maxdepth: 1
   :caption:  Developer notes

   dev/release

******************
Indices and tables
******************

.. toctree::

* :ref:`genindex`

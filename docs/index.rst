.. image:: gwpy-docs-logo.png

.. title:: Docs

.. ifconfig:: '+' in release

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

.. image:: https://badge.fury.io/py/gwpy.svg
    :target: https://badge.fury.io/py/gwpy
    :alt: GWpy PyPI release badge
.. image:: https://zenodo.org/badge/9979119.svg
    :target: https://zenodo.org/badge/latestdoi/9979119
    :alt: GWpy DOI badge
.. image:: https://img.shields.io/pypi/l/gwpy.svg
    :target: https://choosealicense.com/licenses/gpl-3.0/
    :alt: GWpy license badge

First steps
~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   What is GWpy? <overview>
   How do I install GWpy? <install/index>
   Citing GWpy <citing>

Working with data
~~~~~~~~~~~~~~~~~

**Working with interferometer data**

.. toctree::
   :maxdepth: 2

   timeseries/index
   signal/index

**Working with state information and segments**

.. toctree::
   :maxdepth: 2

   segments/index
   timeseries/statevector

**Working with frequency-domain data**

.. toctree::
   :maxdepth: 2

   spectrum/index
   spectrogram/index

**Working with tabular data and events**

.. toctree::
   :maxdepth: 2

   table/index


**Visualising data**

.. toctree::
   :maxdepth: 2

   plot/index
   cli/index

**Other utilities**

.. toctree::
   :maxdepth: 1

   detector/channel
   time/index
   astro/index
   env

**Developer notes**

.. toctree::
   :maxdepth: 1

   dev/release

.. ----------------------------------------------------------------
.. other sections (not directly linked, but need for cross-linking)

.. toctree::
   :hidden:

   examples/index

Indices and tables
~~~~~~~~~~~~~~~~~~

.. toctree::

* :ref:`genindex`

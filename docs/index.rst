.. image:: https://gwpy.github.io/images/gwpy_docs_1200.png

.. title:: Docs

GWpy is a collaboration-driven `Python <http://www.python.org>`_ package
providing tools for studying data from ground-based gravitational-wave
detectors.

GWpy provides a user-friendly, intuitive interface to the common time-domain
and frequency-domain data produced by the `LIGO <http://www.ligo.org>`_ and
`Virgo <http://www.ego-gw.it>`_ instruments and their analysis,
with easy-to-follow tutorials at each step.

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

   time/index
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
   table/index

Visualisation
~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   plotter/index
   cli/index

.. examples
.. ~~~~~~~~

.. toctree::
   :hidden:

   examples/index

Indices and tables
~~~~~~~~~~~~~~~~~~

.. toctree::

* :ref:`genindex`
* :ref:`modindex`

|

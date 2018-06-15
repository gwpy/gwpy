.. currentmodule:: gwpy.frequencyseries

.. _gwpy-frequencyseries-io:

#########################################
Reading and writing frequency-domain data
#########################################

The `FrequencySeries` object includes :meth:`~FrequencySeries.read` and :meth:`~FrequencySeries.write` methods to enable reading from and writing to files respectively.
For example, to read from an ASCII file containing frequency and amplitude columns::

    >>> data = FrequencySeries.read('my-data.txt')

The ``format`` keyword argument can be used to manually identify the input file-format, but is not required where the file extension is sufficiently well understood.

The :meth:`~FrequencySeries.read` and :meth:`~FrequencySeries.write` methods take different arguments and keywords based on the input/output file format, see the following sections for details on reading/writing for each of the built-in formats.
Those formats are:

- :ref:`gwpy-frequencyseries-io-ascii`
- :ref:`gwpy-frequencyseries-io-hdf5`
- :ref:`gwpy-frequencyseries-io-ligolw`

.. _gwpy-frequencyseries-io-ascii:

=====
ASCII
=====

GWpy supports writing `FrequencySeries` data to ASCII in a two-column ``frequency`` and ``amplitude`` format.

Reading
-------

To read a `FrequencySeries` from ASCII::

   >>> t = FrequencySeries.read('data.txt')

See :func:`numpy.loadtxt` for keyword argument options.

Writing
-------

To write a `FrequencySeries` to ASCII::

   >>> t.write('data.txt')

See :func:`numpy.savetxt` for keyword argument options.


.. _gwpy-frequencyseries-io-hdf5:

====
HDF5
====

GWpy allows storing data in HDF5 format files, using a custom specification for storage of metadata.

Reading
-------

To read `FrequencySeries` data held in HDF5 files pass the filename (or filenames) or the source, and the path of the data inside the HDF5 file::

   >>> data = FrequencySeries.read('data.h5', 'psd')

Writing
-------

Data held in a `FrequencySeries` can be written to an HDF5 file via::

   >>> data.write('output.hdf', 'psd')

If the target file already exists, an :class:`~exceptions.IOError` will be raised, use ``overwrite=True`` to force a new file to be written.

To add a `FrequencySeries` to an existing file, use ``append=True``::

    >>> data.write('output.h5', 'psd2', append=True)

To replace an dataset in an existing file, while preserving other data, use *both* ``append=True`` and ``overwrite=True``::

    >>> data.write('output.h5', 'psd', append=True, overwrite=True)


.. _gwpy-frequencyseries-io-ligolw:

===============
``LIGO_LW`` XML
===============

**Additional dependencies:** :mod:`glue.ligolw`

Alongside storing :ref:`tabular data <gwpy-table-io-ligolw>`, the ``LIGO_LW`` XML format allows storing array data.

Reading
-------

A `FrequencySeries` can be read from LIGO_LW XML by passing the file path and the ``Name`` of the containing ``<LIGO_LW>`` element:

   >>> data = FrequencySeries.read('data.xml', 'psd')

Writing
-------

Writing `FrequencySeries` to LIGO_LW XML files is not supported.

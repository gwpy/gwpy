.. currentmodule:: pydischarge.frequencyseries

.. _pydischarge-frequencyseries-io:

#########################################
Reading and writing frequency-domain data
#########################################

The `FrequencySeries` object includes :meth:`~FrequencySeries.read` and :meth:`~FrequencySeries.write` methods to enable reading from and writing to files respectively.
For example, to read from an ASCII file containing frequency and amplitude columns::

    >>> data = FrequencySeries.read('my-data.txt')

The ``format`` keyword argument can be used to manually identify the input file-format, but is not required where the file extension is sufficiently well understood.

The :meth:`~FrequencySeries.read` and :meth:`~FrequencySeries.write` methods take different arguments and keywords based on the input/output file format, see the following sections for details on reading/writing for each of the built-in formats.
Those formats are:

- :ref:`pydischarge-frequencyseries-io-ascii`
- :ref:`pydischarge-frequencyseries-io-hdf5`
- :ref:`pydischarge-frequencyseries-io-ligolw`

.. _pydischarge-frequencyseries-io-ascii:

=====
ASCII
=====

pyDischarge supports writing `FrequencySeries` data to ASCII in a two-column ``frequency`` and ``amplitude`` format.

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


.. _pydischarge-frequencyseries-io-hdf5:

====
HDF5
====

pyDischarge allows storing data in HDF5 format files, using a custom specification for storage of metadata.

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


.. _pydischarge-frequencyseries-io-ligolw:

===============
``LIGO_LW`` XML
===============

**Additional dependencies:** :mod:`python-ligo-lw`

Alongside storing :ref:`tabular data <pydischarge-table-io-ligolw>`, the ``LIGO_LW``
XML format allows storing array data.
These arrays are stored in ``<LIGO_LW>`` elements, which describe the metadata
for an array (name, GPS epoch, instrument, etc.), which contain an
``<Array>`` element that contains the actual data values.

.. note::

   For more information on the format and the parsing library,
   see :mod:`ligo.lw.array`.

Reading
-------

To read a `FrequencySeries` from a ``LIGO_LW`` XML file::

   >>> data = FrequencySeries.read('data.xml')

If the file contains multiple ``<Array>`` elements,
you will have to provide additional keyword arguments to select which
element to use:

.. table:: Keyword arguments for `FrequencySeries.read` with ``LIGO_LW`` format
   :align: left
   :name: frequencyseries-read-ligolw-kwargs

   +------------------+-------+---------+--------------------------------------------+
   | Keyword          | Type  | Default | Usage                                      |
   +==================+=======+=========+============================================+
   | ``name``         | `str` | `None`  | ``Name`` of ``<Array>`` element to read    |
   +------------------+-------+---------+--------------------------------------------+
   | ``epoch``        | `int` | `None`  | GPS value of the ``<Time>`` element that   |
   |                  |       |         | is the *sibling* of the desired            |
   |                  |       |         | ``<Array>``                                |
   +------------------+-------+---------+--------------------------------------------+
   | ``<Param Name>`` |       |         | Other kwargs can be given as the ``Name``  |
   |                  |       |         | of a ``<Param>`` element that is the       |
   |                  |       |         | *sibling* of the desired ``<Array>``,      |
   |                  |       |         | and its value                              |
   +------------------+-------+---------+--------------------------------------------+

For example::

   >>> data = FrequencySeries.read("psd.xml.gz", name="H1")
   >>> data = FrequencySeries.read("psd.xml.gz", epoch=1241492407, f0=0, instrument="H1")

Writing
-------

Writing `FrequencySeries` to ``LIGO_LW`` XML files is not supported, but a
contribution that implements this would be welcomed.

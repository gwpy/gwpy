.. currentmodule:: gwpy.segments

.. _gwpy-segments-io:

##################################
Reading/writing segments and flags
##################################

The `SegmentList`, `DataQualityFlag`, and `DataQualityDict` objects each include :meth:`read` and :meth:`write` methods to enable reading from and writing to a number of different file formats for segment-like objects.
As with other classes in GWpy, the ``format`` keyword argument can be used to manually specify the input or output format, if the file extension isn't obvious enough.

The :meth:`read` and :meth:`write` methods take different arguments and keywords based on the input/output file format, see the following sections for details on reading/writing for each of the built-in formats. Those formats are:

- :ref:`gwpy-segments-io-ligolw`
- :ref:`gwpy-segments-io-hdf5`
- :ref:`gwpy-segments-io-json`

.. _gwpy-segments-io-ligolw:

===============
``LIGO_LW`` XML
===============

**Additional dependencies:** :doc:`igwn-ligolw <igwn-ligolw:index>`

The LIGO Scientific Collaboration uses a custom scheme of XML in which to
store tabular data, called ``LIGO_LW``.
Complementing the scheme is a python library - :doc:`igwn-ligolw <igwn-ligolw:index>` -
which allows users to read and write all of the different types of tabular data
produced by gravitational-wave searches.

Segments are stored in ``LIGO_LW`` format using a trio of tables:

.. table:: ``LIGO_LW`` XML tables for segment storage
    :align: left
    :name: ligolw-segment-tables

    ===================  ==========================================================
    Table name           Contents
    ===================  ==========================================================
    ``segment_definer``  Definitions for each flag, including names and versions
    ``segment_summary``  Known segments for each flag
    ``segment``          Active segments for each flag
    ===================  ==========================================================

.. _gwpy-segments-io-ligolw-reading:

Reading
-------

The :meth:`DataQualityFlag.read` method takes in the file path (or paths) and the name of the flag to read::

    >>> f = DataQualityFlag.read('segments.xml', 'L1:DMT-ANALYSIS_READY:1')

This will parse of each of three tables for references to the given name, returning the metadata and segments as a `DataQualityFlag`. The result may be something like::

    >>> print(f)
    DataQualityFlag('L1:DMT-ANALYSIS_READY:1',
                    known=[[1000000000 ... 1000000100)],
                    active=[[1000000000 ... 1000000034)
                            [1000000065 ... 1000000100)],
                    description=None)

This indicates a single 'known' segment starting at GPS time 1000000000, with two active segments.

**These results are simulated, and do not actually indicate operating times of the LIGO-Livingston observatory.**

The ``coalesce=True`` keyword argument can be used to combine overlapping segments into a single, longer segment.

.. _gwpy-segments-io-ligolw-writing:

Writing
-------

To write a `DataQualityFlag` to file in ``LIGO_LW`` format, use the :meth:`~DataQuality.write` method::

    >>> f.write('new-segments.xml')

As with :ref:`writing tables <gwpy-table-io-ligolw>`, if the target file already exists, an :class:`~exceptions.IOError` will be raised, use ``overwrite=True`` to force a new file to be written.

To write a table to an existing file, use ``append=True``::

    >>> f.write('new-segments.xml', append=True)

To replace the segment tables in an existing file, while preserving other tables, use *both* ``append=True`` and ``overwrite=True``::

    >>> f.write('new-table.xml', append=True, overwrite=True)

Extra attributes can be written to the tables via the ``attrs={}`` keyword, all attributes are set for all three of the segment-related tables::

    >>> f.write('new-table.xml', append=True, overwrite=True, attrs={'process_id': 0})

.. _gwpy-segments-io-ligolw-dict:

`DataQualityDict`
-----------------

The `DataQualityDict` :meth:`DataQualityDict.read` and :meth:`DataQualityDict.write` methods work in an almost identical manner, taking a list of flag names when reading::

    >>> fdict = DataQualityFlag.read('segments.xml', ['H1:DMT-ANALYSIS_READY:1', 'L1:DMT-ANALYSIS_READY:1'])

Identical arguments should be used relative to the :meth:`DataQualityFlag.write` method when writing::

    >>> fdict.write('new-segments.xml')

.. _gwpy-segments-io-hdf5:

====
HDF5
====

GWpy uses HDF5 Groups to store a `DataQualityFlag`, with each of the :attr:`~DataQualityFlag.known` and :attr:`~DataQualityFlag.active` segment lists stored in a Dataset, and extra metadata stored in the Group's attributes.

.. _gwpy-segments-io-hdf5-reading:

Reading
-------

To read a `DataQualityFlag` from an ``HDF5``-format file::

    >>> f = DataQualityFlag.read('segments.hdf')

As with reading other classes from HDF5, the ``path`` keyword should be used to specify the name of the HDF5 group that contains the given flag.

The ``coalesce=True`` keyword can be used to :meth:`~DataQualityFlag.coalesce` the :attr:`~DataQualityFlag.known` and :attr:`~DataQualityFlag.active` segment lists before returning - by default the segments will be returned as read from the file.

.. _gwpy-segments-io-hdf5-writing:

Writing
-------

To write a `DataQualityFlag` to an ``HDF5``-format file::

    >>> f.write('new-segments.hdf5')

As with reading, the ``path`` keyword should be used to specify the name of the HDF5 group to which the given flag should be written.

Alternatively, an HDF5 group can be passed directly to :meth:`~DataQualityFlag.write` when writing multiple objects to the same file.

.. _gwpy-segments-io-hdf5-dict:

`DataQualityDict`
-----------------

As with ``LIGO_LW`` XML, the `DataQualityDict` :meth:`DataQualityDict.read` and :meth:`DataQualityDict.write` methods work in an almost identical manner, taking a list of flag names when reading::

    >>> fdict = DataQualityFlag.read('segments.hdf5', ['H1:DMT-ANALYSIS_READY:1', 'L1:DMT-ANALYSIS_READY:1'])

Identical arguments should be used relative to the :meth:`DataQualityFlag.write` method when writing::

    >>> fdict.write('new-segments.hdf5')

.. _gwpy-segments-io-json:

====
JSON
====

The DQSEGDB server uses JSON as the intermediate format for returning information during queries.

.. _gwpy-segments-io-json-reading:

Reading
-------

To read a `DataQualityFlag` from JSON, simply pass the path of the file::

    >>> f = DataQualityFlag.read('segments.json')

See :func:`json.load` for acceptable keyword arguments options.

.. _gwpy-segments-io-json-writing:

Writing
-------

To write a `DataQualityFlag` to JSON::

    >>> f = DataQualityFlag.write('new-segments.json')

See :func:`json.dump` for keyword arguments options.

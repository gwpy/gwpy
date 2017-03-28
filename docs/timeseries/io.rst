.. currentmodule:: gwpy.timeseries
.. include:: ../references.txt

.. _gwpy-timeseries-io:

####################################
Reading and writing time-domain data
####################################

The `TimeSeries` object includes :meth:`~TimeSeries.read` and :meth:`~TimeSeries.write` methods to enable reading from and writing to files respectively.
For example, to read from an ASCII file containing time and amplitude columns::

   >>> data = TimeSeries.read('my-data.txt')

The ``format`` keyword argument can be used to manually identify the input file-format, but is not required where the file extension is sufficiently well understood.

The :meth:`~TimeSeries.read` and :meth:`~TimeSeries.write` methods take different arguments and keywords based on the input/output file format, see the following sections for details on reading/writing for each of the built-in formats.
Those formats are:

- :ref:`gwpy-timeseries-io-ascii`
- :ref:`gwpy-timeseries-io-gwf`
- :ref:`gwpy-timeseries-io-hdf5`

.. _gwpy-timeseries-io-ascii:

=====
ASCII
=====

GWpy supports writing `TimeSeries` (and `~gwpy.frequencyseries.FrequencySeries`) data to ASCII in a two-column ``time`` and ``amplitude`` format.

Reading
-------

To read a `TimeSeries` from ASCII::

   >>> t = TimeSeries.read('data.txt')

See :func:`numpy.loadtxt` for keyword argument options.

Writing
-------

To write a `TimeSeries` to ASCII::

   >>> t.write('data.txt')

See :func:`numpy.savetxt` for keyword argument options.

.. _gwpy-timeseries-io-gwf:

===============================
Gravitational-wave frames (GWF)
===============================

**Additional dependencies:**  |LDAStools.frameCPP|_ or |lalframe|_

The raw observatory data are archived in ``.gwf`` files, a custom binary format that efficiently stores the time streams and all necessary metadata, for more details about this particular data format, take a look at the specification document `LIGO-T970130 <https://dcc.ligo.org/LIGO-T970130/public>`_.

Reading
-------

To read data from a GWF file, pass the input file path (or paths) and the name of the data channel to read::

    >>> data = TimeSeries.read('HLV-GW100916-968654552-1.gwf', 'L1:LDAS-STRAIN')

.. note::

   The ``HLV-GW100916-968654552-1.gwf`` file is included with the GWpy source under `/gwpy/tests/data/ <https://github.com/gwpy/gwpy/tree/master/gwpy/tests/data>`_.

Reading a `StateVector` uses the same syntax::

    >>> data = StateVector.read('my-state-data.gwf', 'L1:GWO-STATE_VECTOR')

Multiple files can be read by passing a list of files::

   >>> data = TimeSeries.read([file1, file2], 'L1:LDAS-STRAIN')

When reading multiple files, the ``nproc`` keyword argument can be used to distribute the reading over multiple CPUs, which should make it faster::

   >>> data = TimeSeries.read([file1, file2, file3, file4], 'L1:LDAS-STRAIN', nproc=2)

The above command will separate the input list of 4 file paths into two sets of 2 files, combining the results into a single `TimeSeries` before returning.

The ``start`` and ``end`` keyword arguments can be used to downselect data to a specific ``[start, end)`` time segment when reading::

    >>> data = TimeSeries.read('HLV-GW100916-968654552-1.gwf', 'L1:LDAS-STRAIN', start=968654552.5, end=968654553)

Additionally, the following keyword arguments can be passed to manipulate the data on-the-fly when reading:

============  =======  ==========================================
Keyword       Type     Usage
============  =======  ==========================================
``resample``  `float`  resample the data to a different number of
                       samples per second
``dtype``     `type`   cast the input data to a different data type
============  =======  ==========================================

For example::

   >>> data = TimeSeries.read('HLV-GW100916-968654552-1.gwf', 'L1:LDAS-STRAIN', resample=2048)

Reading multiple channels
-------------------------

To read multiple channels from one or more GWF files (rather than opening and closing the files multiple times), use the `TimeSeriesDict` or `StateVectorDict` classes, and pass a list of data channel names::

    >>> data = TimeSeriesDict.read('HLV-GW100916-968654552-1.gwf', ['H1:LDAS-STRAIN', 'L1:LDAS-STRAIN'])

In this case the ``resample`` and ``dtype`` keywords can be given as a single value used for all data channels, or a `dict` mapping an argument for each data channel name::

    >>> data = TimeSeriesDict.read('HLV-GW100916-968654552-1.gwf', ['H1:LDAS-STRAIN', 'L1:LDAS-STRAIN'], resample={'H1:LDAS-STRAIN': 2048})

The above example will resample only the ``'H1:LDAS-STRAIN'`` `TimeSeries` and will not modify that for ``'L1:LDAS-STRAIN'``.

.. note::

   A mix of `TimeSeries` and `StateVector` objects can be read by using only `TimeSeriesDict` class, and casting the returned data to a `StateVector` using :meth:`~TimeSeries.view`.

Writing
-------

To write data held in any of the :mod:`gwpy.timeseries` classes to a GWF file, simply use::

    >>> data.write('output.gwf')

**If the output file already exists it will be overwritten.**

.. _gwpy-timeseries-io-hdf5:

====
HDF5
====

**Additional dependencies:** |h5py|_

GWpy allows storing data in HDF5 format files, using a custom specification for storage of metadata.

.. warning::

   Reading/writing multiple datasets with the `TimeSeriesDict` and `StateVectorDict` classes is not supported at this time.

Reading
-------

To read `TimeSeries` or `StateVector` data held in HDF5 files pass the filename (or filenames) or the source, and the path of the data inside the HDF5 file::

   >>> data = TimeSeries.read('HLV-GW100916-968654552-1.hdf', 'L1:LDAS-STRAIN')

As with GWF, the ``start`` and ``end`` keyword arguments can be used to downselect data to a specific ``[start, end)`` time segment when reading::

    >>> data = TimeSeries.read('HLV-GW100916-968654552-1.hdf', 'L1:LDAS-STRAIN', start=968654552.5, end=968654553)

Writing
-------

Data held in a `TimeSeries` or `StateVector` can be written to an HDF5 file via::

   >>> data.write('output.hdf')

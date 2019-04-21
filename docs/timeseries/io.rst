.. currentmodule:: gwpy.timeseries

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
- :ref:`gwpy-timeseries-io-wav`

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

    >>> data = TimeSeries.read('HLV-HW100916-968654552-1.gwf', 'L1:LDAS-STRAIN')

.. note::

   The ``HLV-HW100916-968654552-1.gwf`` file is included with the GWpy source under `/gwpy/testing/data/ <https://github.com/gwpy/gwpy/tree/master/gwpy/testing/data>`_.

Reading a `StateVector` uses the same syntax::

    >>> data = StateVector.read('my-state-data.gwf', 'L1:GWO-STATE_VECTOR')

Multiple files can be read by passing a list of files::

   >>> data = TimeSeries.read([file1, file2], 'L1:LDAS-STRAIN')

When reading multiple files, the ``nproc`` keyword argument can be used to distribute the reading over multiple CPUs, which should make it faster::

   >>> data = TimeSeries.read([file1, file2, file3, file4], 'L1:LDAS-STRAIN', nproc=2)

The above command will separate the input list of 4 file paths into two sets of 2 files, combining the results into a single `TimeSeries` before returning.

The ``start`` and ``end`` keyword arguments can be used to downselect data to a specific ``[start, end)`` time segment when reading::

    >>> data = TimeSeries.read('HLV-HW100916-968654552-1.gwf', 'L1:LDAS-STRAIN', start=968654552.5, end=968654553)

Additionally, the following keyword arguments can be used:

.. table:: Keyword arguments for `TimeSeries.read`
   :align: left
   :name: timeseries-read-kwargs

   ============  =======  =======  ==========================================
   Keyword       Type     Default  Usage
   ============  =======  =======  ==========================================
   ``scaled``    `bool`   `True`   Apply ADC calibration when reading
   ``type``      `str`    `None`   `dict` of channel types
                                   (``'ADC'``, ``'Proc'``, or ``'Sim'``) for
                                   each channel to be read. This option
                                   optimises the reading operation.
   ============  =======  =======  ==========================================

Reading multiple channels
-------------------------

To read multiple channels from one or more GWF files (rather than opening and closing the files multiple times), use the `TimeSeriesDict` or `StateVectorDict` classes, and pass a list of data channel names::

    >>> data = TimeSeriesDict.read('HLV-HW100916-968654552-1.gwf', ['H1:LDAS-STRAIN', 'L1:LDAS-STRAIN'])

In this case the ``resample`` and ``dtype`` keywords can be given as a single value used for all data channels, or a `dict` mapping an argument for each data channel name::

    >>> data = TimeSeriesDict.read('HLV-HW100916-968654552-1.gwf', ['H1:LDAS-STRAIN', 'L1:LDAS-STRAIN'],
    ...                            resample={'H1:LDAS-STRAIN': 2048})

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

GWpy allows storing data in HDF5 format files, using a custom specification for storage of metadata.

.. warning::

   To read GWOSC (LOSC) data from HDF5, please see
   :ref:`gwpy-timeseries-io-hdf5-gwosc`.

Reading
-------

To read `TimeSeries` or `StateVector` data held in HDF5 files pass the filename (or filenames) or the source, and the path of the data inside the HDF5 file::

   >>> data = TimeSeries.read('HLV-HW100916-968654552-1.hdf', 'L1:LDAS-STRAIN')

As with GWF, the ``start`` and ``end`` keyword arguments can be used to downselect data to a specific ``[start, end)`` time segment when reading::

    >>> data = TimeSeries.read('HLV-HW100916-968654552-1.hdf', 'L1:LDAS-STRAIN', start=968654552.5, end=968654553)

Analogously to GWF, you can read multiple `TimeSeries` from an HDF5 file via :meth:`TimeSeriesDict.read`::

   >>> data = TimeSeriesDict.read('HLV-HW100916-968654552-1.hdf')

By default, all matching datasets in the file will be read, to restrict the output, specify the names of the datasets you want::

   >>> data = TimeSeriesDict.read('HLV-HW100916-968654552-1.hdf', ['H1:LDAS-STRAIN', 'L1:LDAS-STRAIN'])


Writing
-------

Data held in a `TimeSeries`, `TimeSeriesDict, `StateVector`, or `StateVectorDict` can be written to an HDF5 file via::

   >>> data.write('output.hdf')

The output argument (``'output.hdf'``) can be a file path, an open `h5py.File` object, or a `h5py.Group` object, to append data to an existing file.

If the target file already exists, an :class:`~exceptions.IOError` will be raised, use ``overwrite=True`` to force a new file to be written.

To write a `TimeSeries` to an existing file, use ``append=True``::

    >>> data.write('output.hdf', append=True)

To replace an existing dataset in an existing file, while preserving other data, use *both* ``append=True`` and ``overwrite=True``::

    >>> data.write('output.hdf', append=True, overwrite=True)

.. _gwpy-timeseries-io-hdf5-gwosc:

============
HDF5 (GWOSC)
============

|GWOSC|_ write data in HDF5 using a custom schema that is incompatible
with `format='hdf5'`.

-------
Reading
-------

GWpy can read data from GWOSC (LOSC) HDF5 files using the `format='hdf5.losc'`
keyword::

   >>> data = TimeSeries.read('H-H1_GWOSC_16KHZ_R1-1187056280-4096.hdf5',
   ...                        format='hdf5.losc')

By default, `TimeSeries.read` will return the contents of the
``/strain/Strain`` dataset, while `StateVector.read` will return those of
``/quality/simple``.

As with regular HDF5, the ``start`` and ``end`` keyword arguments can be used to downselect data to a specific ``[start, end)`` time segment when reading.


.. _gwpy-timeseries-io-wav:

===
WAV
===

Any `TimeSeries` can be written to / read from a WAV file using :meth:`TimeSeries.read`:

.. warning::

   No metadata are stored in the WAV file except the sampling rate, so any units or GPS timing information are lost when converting to/from WAV.

Reading
-------

To read a `TimeSeries` from WAV::

   >>> t = TimeSeries.read('data.wav')

See :func:`scipy.io.wavfile.read` for any keyword argument options.

Writing
-------

To write a `TimeSeries` to WAV::

   >>> t.write('data.wav')

See :func:`scipy.io.wavfile.write` for keyword argument options.

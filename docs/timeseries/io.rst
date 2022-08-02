:tocdepth: 3

.. currentmodule:: gwpy.timeseries

.. _gwpy-timeseries-io:

####################################
Reading and writing time series data
####################################

The `TimeSeries` object includes :meth:`~TimeSeries.read` and
:meth:`~TimeSeries.write` methods to enable reading from and writing to files
respectively.
For example, to read from an ASCII file containing time and amplitude columns:

.. code-block:: python
   :name: gwpy-timeseries-io-example

   >>> from gwpy.timeseries import TimeSeries
   >>> data = TimeSeries.read('my-data.txt')

:meth:`TimeSeries.read` will attempt to automatically identify the file
format based on the file extension and/or the contents of the file, however,
the ``format`` keyword argument can be used to manually identify the input
file-format.

The :meth:`~TimeSeries.read` and :meth:`~TimeSeries.write` methods take
different arguments and keywords based on the input/output file format,
see :ref:`gwpy-timeseries-io-formats` for details on reading/writing for
each of the built-in formats.

.. _gwpy-timeseries-io-discovery:

***************************************
Automatic discovery of GW detector data
***************************************

================
GW detector data
================

Gravitational-wave detector data, including all engineering diagnostic data
as well as the calibrated 'strain' data that are searched for GW signals,
are archived in :ref:`GWF <gwpy-timeseries-io-gwf>` files stored at the
relevant observatory.
These data are available locally to authenticated users of the associated
computing centres (typically collaboration members), but are also
distributed using |CVMFS|_ and are available remotely using |nds2|_.
Access to these data is restricted to active collaboration members.

Additionally |GWOSCl|_ hosts publicly-accessible 'open' data, with *event*
datasets made available at the same time as the relevant result publication
and typically including ~1 hour of data around each published event detection,
and *bulk* datasets with the entire observing run data available roughly
18 months after the end of the run.

GWOSC also hosts the |GWOSC_AUX_RELEASE|_, providing public access to
environmental sensor data around |GW170814|_.
These data are freely accessible using |nds2|_.

======================
Data discovery methods
======================

.. toctree::

   opendata
   datafind

.. _gwpy-timeseries-io-formats:

*********************
Built-in file formats
*********************

.. _gwpy-timeseries-io-ascii:

=====
ASCII
=====

GWpy supports writing `TimeSeries` (and `~gwpy.frequencyseries.FrequencySeries`) data to ASCII in a two-column ``time`` and ``amplitude`` format.

Reading
-------

To read a `TimeSeries` from ASCII:

.. code-block:: python

   >>> t = TimeSeries.read('data.txt')

See :func:`numpy.loadtxt` for keyword argument options.

Writing
-------

To write a `TimeSeries` to ASCII:

.. code-block:: python

   >>> t.write('data.txt')

See :func:`numpy.savetxt` for keyword argument options.

.. _gwpy-timeseries-io-gwf:

===
GWF
===

**Additional dependencies:**  |LDAStools.frameCPP|_ or |framel|_ or |lalframe|_

The raw observatory data are archived in ``.gwf`` files, a custom binary format that efficiently stores the time streams and all necessary metadata, for more details about this particular data format, take a look at the specification document `LIGO-T970130 <https://dcc.ligo.org/LIGO-T970130/public>`_.

Reading
-------

To read data from a GWF file, pass the input file path (or paths) and the name of the data channel to read:

.. code-block:: python

   >>> data = TimeSeries.read('HLV-HW100916-968654552-1.gwf', 'L1:LDAS-STRAIN')

.. note::

   The ``HLV-HW100916-968654552-1.gwf`` file is included with the GWpy source under `/gwpy/testing/data/ <https://github.com/gwpy/gwpy/tree/main/gwpy/testing/data>`_.

Reading a `StateVector` uses the same syntax:

.. code-block:: python

   >>> data = StateVector.read('my-state-data.gwf', 'L1:GWO-STATE_VECTOR')

Multiple files can be read by passing a list of files:

.. code-block:: python

   >>> data = TimeSeries.read([file1, file2], 'L1:LDAS-STRAIN')

When reading multiple files, the ``nproc`` keyword argument can be used to distribute the reading over multiple CPUs, which should make it faster:

.. code-block:: python

   >>> data = TimeSeries.read([file1, file2, file3, file4], 'L1:LDAS-STRAIN', nproc=2)

The above command will separate the input list of 4 file paths into two sets of 2 files, combining the results into a single `TimeSeries` before returning.

The ``start`` and ``end`` keyword arguments can be used to downselect data to a specific ``[start, end)`` time segment when reading:

.. code-block:: python

   >>> data = TimeSeries.read('HLV-HW100916-968654552-1.gwf', 'L1:LDAS-STRAIN', start=968654552.5, end=968654553)

Additionally, the following keyword arguments can be used:

.. warning::

   These keyword arguments are only supported when using the
   |LDASTools.frameCPP|_ GWF API.

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

To read multiple channels from one or more GWF files (rather than opening and closing the files multiple times), use the `TimeSeriesDict` or `StateVectorDict` classes, and pass a list of data channel names:

.. code-block:: python

   >>> data = TimeSeriesDict.read('HLV-HW100916-968654552-1.gwf', ['H1:LDAS-STRAIN', 'L1:LDAS-STRAIN'])

.. note::

   A mix of `TimeSeries` and `StateVector` objects can be read by using only `TimeSeriesDict` class, and casting the returned data to a `StateVector` using :meth:`~TimeSeries.view`.

Writing
-------

To write data held in any of the :mod:`gwpy.timeseries` classes to a GWF file, simply use:

.. code-block:: python

   >>> data.write('output.gwf')

**If the output file already exists it will be overwritten.**

GWF library availability
------------------------

(Last we checked...) The three GWF library interfaces are available on
the following platforms:

.. table:: GWF library availability by platform
   :align: left
   :name: gwf-library-platforms

   =====================  ==============================  =====  =====  =======
   Library                Conda-forge package name        Linux  macOS  Windows
   =====================  ==============================  =====  =====  =======
   |LDASTools.frameCPP|_  ``python-ldas-tools-framecpp``  Yes    Yes    No
   |framel|_              ``python-framel``               Yes    Yes    Yes
   |lalframe|_            ``python-lalframe``             Yes    Yes    No
   =====================  ==============================  =====  =====  =======

.. _gwpy-timeseries-io-hdf5:

====
HDF5
====

GWpy allows storing data in HDF5 format files, using a custom specification for storage of metadata.

.. warning::

   To read GWOSC data from HDF5, please see
   :ref:`gwpy-timeseries-io-hdf5-gwosc`.

Reading
-------

To read `TimeSeries` or `StateVector` data held in HDF5 files pass the filename (or filenames) or the source, and the path of the data inside the HDF5 file:

.. code-block:: python

   >>> data = TimeSeries.read('HLV-HW100916-968654552-1.hdf', 'L1:LDAS-STRAIN')

As with GWF, the ``start`` and ``end`` keyword arguments can be used to downselect data to a specific ``[start, end)`` time segment when reading:

.. code-block:: python

   >>> data = TimeSeries.read('HLV-HW100916-968654552-1.hdf', 'L1:LDAS-STRAIN', start=968654552.5, end=968654553)

Analogously to GWF, you can read multiple `TimeSeries` from an HDF5 file via :meth:`TimeSeriesDict.read`:

.. code-block:: python

   >>> data = TimeSeriesDict.read('HLV-HW100916-968654552-1.hdf')

By default, all matching datasets in the file will be read, to restrict the output, specify the names of the datasets you want:

.. code-block:: python

   >>> data = TimeSeriesDict.read('HLV-HW100916-968654552-1.hdf', ['H1:LDAS-STRAIN', 'L1:LDAS-STRAIN'])


Writing
-------

Data held in a `TimeSeries`, `TimeSeriesDict, `StateVector`, or `StateVectorDict` can be written to an HDF5 file via:

.. code-block:: python

   >>> data.write('output.hdf')

The output argument (``'output.hdf'``) can be a file path, an open `h5py.File` object, or a `h5py.Group` object, to append data to an existing file.

If the target file already exists, an :class:`~exceptions.IOError` will be raised, use ``overwrite=True`` to force a new file to be written.

To write a `TimeSeries` to an existing file, use ``append=True``:

.. code-block:: python

   >>> data.write('output.hdf', append=True)

To replace an existing dataset in an existing file, while preserving other data, use *both* ``append=True`` and ``overwrite=True``:

.. code-block:: python

   >>> data.write('output.hdf', append=True, overwrite=True)

.. _gwpy-timeseries-io-hdf5-gwosc:

============
HDF5 (GWOSC)
============

|GWOSC|_ write data in HDF5 using a custom schema that is incompatible
with `format='hdf5'`.

Reading
-------

GWpy can read data from GWOSC HDF5 files using the `format='hdf5.gwosc'`
keyword:

.. code-block:: python

   >>> data = TimeSeries.read(
   ...     "H-H1_GWOSC_16KHZ_R1-1187056280-4096.hdf5",
   ...     format="hdf5.gwosc",
   ... )

By default, :meth:`TimeSeries.read` will return the contents of the
``/strain/Strain`` dataset, while :meth:`StateVector.read` will return those
of ``/quality/simple``.

As with regular HDF5, the ``start`` and ``end`` keyword arguments can be used
to downselect data to a specific ``[start, end)`` time segment when reading.


.. _gwpy-timeseries-io-wav:

===
WAV
===

Any `TimeSeries` can be written to / read from a WAV file using :meth:`TimeSeries.read`:

.. warning::

   No metadata are stored in the WAV file except the sampling rate, so any units or GPS timing information are lost when converting to/from WAV.

Reading
-------

To read a `TimeSeries` from WAV:

.. code-block:: python

   >>> t = TimeSeries.read('data.wav')

See :func:`scipy.io.wavfile.read` for any keyword argument options.

Writing
-------

To write a `TimeSeries` to WAV:

.. code-block:: python

   >>> t.write('data.wav')

See :func:`scipy.io.wavfile.write` for keyword argument options.

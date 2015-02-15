Customizing time series plots
=============================

The time series plots label the axes X and Y.  To adjust the limits use the ``--xmin``, ``xmax``,
``ymin``, ``ymax``.  The default is to set the axes limits to show all the data.  A good
reason to adjust the y-axis limits is to compare different plots.  The x-axis is generally controlled
by the amount of data transferred but the data transfer is only controllable to integer second resolution.
The x-axis limits may be set in fractional seconds of arbitrary resolution.  The x limits are
specified in GPS seconds, the same as start time.

To produce a log-log or semi-log plot use the ``logx`` and/or ``logy`` parameters.

The following image is a semi-log plot of the pressure in HVEA during a pump down procedure.  It
shows 10 hours of the mean minute trend.  Notice the syntax for that channel.
It is created with the following command line

.. code-block:: sh

    gwpy-ldvw.py timeseries --chan LVE-EX:X3_810BTORR.mean,m-trend --start 1072539420 \
        --dur 36000 --logy

.. image:: /../../cli_examples/cli-ts-02.png
    :align: center
    :alt: HVEA pressure during pump down

The next example is a sub-second plot of a DARM channel.
It is created with the following command line:

.. code-block:: sh

    gwpy-ldvw.py timeseries --chan L1:OAF-CAL_DARM_DQ --start 1107936100 --xmin 1107936100.2 \
        --xmax 1107936100.25 --epoch 0.225 --duration 1


.. image:: /../../cli_examples/cli-ts-01.png
    :align: center
    :alt: sample spectrum

Notice the use of the ``--epoch`` parameter to set the 0 point of the time axis to the middle
of the plot.
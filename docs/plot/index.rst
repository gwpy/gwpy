.. currentmodule:: gwpy.plot

.. _gwpy-plot:

###################################
Plotting in GWpy (:mod:`gwpy.plot`)
###################################

The :mod:`gwpy.plot` module provides integrated extensions to the fantastic
data visualisation tools provided by |matplotlib|_.

=============================================
Basic plotting with :mod:`~matplotlib.pyplot`
=============================================

Each of the data representations provided by `gwpy` can be directly passed
to the standard methods available in `~matplotlib.pyplot`:

.. plot::
   :include-source:

   >>> from gwpy.timeseries import TimeSeries
   >>> data = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)
   >>> from matplotlib import pyplot as plt
   >>> plt.plot(data)
   >>> plt.show()

==============================
:meth:`.plot` instance methods
==============================

Each of the data representations provided by `gwpy` also come with a
:meth:`.plot` method that provides a figure with improved defaults tailored
to those data:

.. plot::
   :include-source:

   >>> from gwpy.timeseries import TimeSeries
   >>> data = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)
   >>> plot = data.plot()
   >>> plot.show()

###########################################################################################
The spectrogram (:class:`gwpy.spectrogram.Spectrogram <gwpy.spectrogram.core.Spectrogram>`)
###########################################################################################

.. currentmodule:: gwpy.spectrogram.core

.. code-block:: python

   >>> from gwpy.spectrogram import Spectrogram

While the :class:`~gwpy.timeseries.core.TimeSeries` allows us to study how the amplitude of a signal changes over time, and the :class:`~gwpy.spectrum.core.Spectrum` allows us to study how that amplitude changes over frequency, the time-frequency :class:`~gwpy.spectrogram.core.Spectrogram` allows us to track the evolution of the `Spectrum` over time.

This object is a 2-dimensional array, essentially a stacked set of spectra, one per unit time.

As always, a `Spectrogram` can be generated from any abitrary data sequence, but here the required metadata are a combination of those required for the :class:`~gwpy.timeserises.core.TimeSeries` and :class:`~gwpy.spectrum.core.Spectrum`::

    >>> import numpy
    >>> specgram = Spectrogram(numpy.random.random((100, 1000)), epoch=1000000000, sample_rate=1, f0=0, df=1)
    >>> print(specgram)
    Spectrogram([[ 0.58030742  0.94586261  0.79559404 ...,  0.25253688  0.61626489
                   0.22785403]
                 [ 0.95930736  0.93154594  0.13234058 ...,  0.13920997  0.94432426
                   0.29442085]
                 [ 0.66572174  0.77702177  0.8900096  ...,  0.18828231  0.81440898
                   0.97455031]
                 ...,
                 [ 0.46696636  0.72475187  0.17941277 ...,  0.19095158  0.83843501
                   0.92154324]
                 [ 0.81492468  0.01945053  0.77665596 ...,  0.73642962  0.78723728
                   0.20995951]
                 [ 0.35161785  0.79137264  0.50710421 ...,  0.39068193  0.61551753
                   0.74846848]],
                name: None,
                unit: None,
                epoch: 2011-09-14 01:46:59.000,
                dt: 1 s,
                f0: 0 Hz,
                df: 1 Hz,
                logf: False)

The full set of metadata that can be provided is as follows:

.. autosummary::

   ~Spectrogram.name
   ~Spectrogram.unit
   ~Spectrogram.epoch
   ~Spectrogram.dt
   ~Spectrogram.f0
   ~Spectrogram.df

============================================================================
Calculating a `Spectrogram` from a :class:`~gwpy.timeseries.core.TimeSeries`
============================================================================

.. currentmodule:: gwpy.timeseries.core

The time-frequency :class:`~gwpy.spectrogram.core.Spectrogram` of a `TimeSeries` can be calculated using the :meth:`TimeSeries.spectrogram`.
We can extend the original :ref:`time-series example <timeseries-plot>` with calculation of a :class:`~gwpy.spectrogram.core.Spectrogram` with a 20-second stride:

.. literalinclude:: spectrogram_plot.py
   :lines: 1-3

========================
Plotting a `Spectrogram`
========================

.. currentmodule:: gwpy.spectrogram.core

Like the :class:`TimeSeries` and :class:`Spectrum`, the `Spectrogram` has a convenient :meth:`~Spectrogram.plot` method, allowing us to view the data.
We can extend the above example to include a plot:

.. plot:: spectrogram/spectrogram_plot.py
   :include-source:

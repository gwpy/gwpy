======================================================================================
GWpy Example: comparing a channel's :class:`~gwpy.spectrum.Spectrum` between two times
======================================================================================

.. currentmodule:: gwpy.spectrum

Problem
-------

I'm interested in comparing the amplitude spectrum of a channel between a
known 'good' time - where the spectrum is what we expect it to be - and a
known 'bad' time - where some excess noise appeared and the spectrum
changed appreciably.

Solution
--------

First, we define what the 'good' and 'bad' times are:

.. literalinclude:: ../../examples/gw_ex_compare_spectra.py
   :lines: 39,40

and how long we want to search each one

.. literalinclude:: ../../examples/gw_ex_compare_spectra.py
   :lines: 41

Then we can fetch the :class:`~gwpy.timeseries.TimeSeries` data for both times for the in-loop photodiode signal for the intensity stabilisation servo (ISS) of the pre-stabilised laser (PSL):

.. literalinclude:: ../../examples/gw_ex_compare_spectra.py
   :lines: 32,44-47

We can now calculate the amplitude spectral density (ASD) for each time using the :meth:`~gwpy.timeseries.TimeSeries.asd` method,

.. literalinclude:: ../../examples/gw_ex_compare_spectra.py
   :lines: 50,51

and make a comparison plot:

.. literalinclude:: ../../examples/gw_ex_compare_spectra.py
   :append: plot.show()
   :lines: 54-58

.. plot:: ../examples/gw_ex_compare_spectra.py

The extra peak at 620 Hz is worrying, so we can zoom in around that frequency range to see what's going on:

.. literalinclude:: gw_ex_compare_spectra_zoom.py
   :append: plot.refresh()
   :lines: 2-5

.. plot:: examples/gw_ex_compare_spectra_zoom.py

That needs investigating, better call it in!

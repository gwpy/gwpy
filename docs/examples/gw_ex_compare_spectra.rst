.. currentmodule:: gwpy.spectrum.core
===========================================================
GWpy.Ex: comparing a Channel's `Spectrum` between two times
===========================================================

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
   :lines: 32,40-41

and how long we want to search each one

.. literalinclude:: ../../examples/gw_ex_compare_spectra.py
   :lines: 42

Then we can fetch the :class:`~gwpy.timeseries.core.TimeSeries` data for both times for the in-loop photodiode signal for the intensity stabilisation servo (ISS) of the pre-stabilised laser (PSL):

.. literalinclude:: ../../examples/gw_ex_compare_spectra.py
   :lines: 33,45-48

We can now calculate the amplitude spectral density (ASD) for each time using the :meth:`~gwpy.timeseries.core.TimeSeries.asd` method,

.. literalinclude:: ../../examples/gw_ex_compare_spectra.py
   :lines: 51-52

and make a comparison plot:

.. literalinclude:: ../../examples/gw_ex_compare_spectra.py
   :lines: 55-58

.. plot:: ../examples/gw_ex_compare_spectra.py

The extra peak at 620 Hz is worrying, so we can zoom in around that frequency range to see what's going on:

.. code:: python

   plot.logx = False
   plot.xlim = [600, 640]
   plot.refresh()

.. plot:: ../examples/gw_ex_compare_spectra_zoom.py

That needs investigating, better call it in!

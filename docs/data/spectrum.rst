####################################
Spectral data (`gwpy.data.Spectrum`)
####################################

Reading spectra from a file
===========================

Spectral data can be read directly from GWF files as follows::

    >>> from gwpy.data import Spectrum
    >>> psd = Spectrum.read('myspectrum.gwf', 'H1:LDAS-STRAIN')

where the two arguments to the `read` function are the path to the GWF file, and the name of the spectrum as written in that file.

Each `~gwpy.data.Spectrum` records the spectral `data` along with the following metadata:

  - `f0`: the starting frequency of the spectrum
  - `df`: the frequency resolution (in Hertz)
  - `units`: the units of the data,
  - `name`: the name for this `~gwpy.data.Spectrum`
  - `logspace`: a boolean flag indicating whether the frequency array should be in a logarithmic scale (default `False`).

Calculating a spectrum
======================

The (average) spectrum for any `~gwpy.data.TimeSeries` can be calculated using one of the `~gwpy.data.TimeSeries.psd` (power spectral density) or `~gwpy.data.TimeSeries.asd` (amplitude spectral density) methods, for example::

    >>> from gwpy.data import TimeSeries
    >>> hoft = TimeSeries.read('G-G1_RDS_C01_L3-1049587200-60.gwf', 'G1:DER_DATA_H')
    >>> hoff = hoft.asd(method='welch', fft_length=2, overlap=1)

where the result is an average spectrum, using one of the `Bartlett`, `Welch` or `median-mean` average spectrum methods.
Here `fft_length` and `overlap` are the lengths (in seconds) of each Fourier transform and the overlap between successive transforms respectively.


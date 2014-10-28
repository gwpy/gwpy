.. currentmodule:: gwpy.timeseries.core

##################################
Reading publicly-available GW data
##################################

The LIGO and Virgo scientific collaborations have released a small number of data sets associated with important milestones in their search for gravitational waves.
These data can be read about and downloaded `here <http://www.ligo.org/science/data-releases.php>`_.

The rest of this page gives an example of how you can use GWpy to read and visualise publicly available data.

========
GW100916
========

GW100916 is the designation given to an interesting event seen in the data from the detector network operating in September 2010.
The gravitational-wave signal found in these data was revealed to be a 'blind injection', a fake signal added to the data without knowledge of the analysis teams, allowing them to test both the detector and the analysis software.
The data for candidate event GW100916 are available `here <http://www.ligo.org/science/GW100916/index.php>`_.

The strain time-series can be downloaded by selecting links under the 'STRAIN DATA H(T)' section, as txt files containing 10 seconds of data for each instrument.
Anybody can download, read, and plot the data entirely in python as follows:

.. literalinclude:: ../../examples/timeseries/public.py
   :lines: 31-33,36,40-41,44-47

.. plot:: ../examples/timeseries/public.py

|

This code is an extract from the full example on :doc:`plotting public LIGO data <../examples/timeseries/public>`.

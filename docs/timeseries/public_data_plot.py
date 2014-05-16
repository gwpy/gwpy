from urllib2 import urlopen
from numpy import asarray
from gwpy.timeseries import TimeSeries
data = urlopen('http://www.ligo.org/science/GW100916/'
               'L-strain_hp30-968654552-10.txt').read()
ts = TimeSeries(asarray(data.splitlines(), dtype=float),
                epoch=968654552, sample_rate=16384)
plot = ts.plot()
plot.set_ylabel('Gravitational-wave strain amplitude')
plot.set_title('LIGO Livingston Observatory data for GW100916')
plot.show()

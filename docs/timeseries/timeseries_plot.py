from gwpy.timeseries import TimeSeries
gwdata = TimeSeries.fetch('H1:LDAS-STRAIN', 'September 16 2010 06:40',
                          'September 16 2010 06:50')
plot = gwdata.plot()
ax = plot.gca()
ax.set_ylabel('Gravitational-wave strain amplitude')
plot.show()

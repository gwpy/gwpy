from gwpy.timeseries import TimeSeries
gwdata = TimeSeries.fetch('H1:LDAS-STRAIN', 968654552, 968654562)
plot = gwdata.plot()
ax = plot.gca()
ax.set_ylabel('Gravitational-wave strain amplitude')
plot.show()

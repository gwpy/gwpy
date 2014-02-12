from gwpy.timeseries import TimeSeries
gwdata = TimeSeries.fetch('H1:LDAS-STRAIN', 'September 16 2010 06:40', 'September 16 2010 06:50')
spectrum = gwdata.asd(8, 4)
plot = spectrum.plot()
ax = plot.gca()
ax.set_xlim(40, 4000)
ax.set_ylabel(r'Gravitational-wave strain ASD [strain$/\sqrt{\mathrm{Hz}}$]')
ax.set_ylim(1e-23, 1e-19)
plot.show()
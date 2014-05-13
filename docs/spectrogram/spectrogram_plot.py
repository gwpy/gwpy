from gwpy.timeseries import TimeSeries
gwdata = TimeSeries.fetch('H1:LDAS-STRAIN', 'September 16 2010 06:40',
                          'September 16 2010 06:50')
specgram = gwdata.spectrogram(20, fftlength=8, overlap=4) ** (1/2.)
plot = specgram.plot(norm='log', vmin=1e-23, vmax=1e-19)
ax = plot.gca()
ax.set_ylim(40, 4000)
ax.set_yscale('log')
plot.add_colorbar(label='Gravitational-wave strain [m/$\sqrt{\mathrm{Hz}}$]')
plot.show()


## GWpy.Ex: plotting a time-series

### Problem

# I would like to study the gravitational wave strain time-series around the time of an interesting simulated signal during the last science run (S6). I have access to the frame files on the LIGO Data Grid machine `ldas-pcdev2.ligo-wa.caltech.edu` and so can read them directly.

### Solution

# First up, we need to decide what times we want. The interesting signal happened between 06:42 an 06:43 on September 16 2010, and so we can set the times from there:


from gwpy.time import Time
start = Time('2010-09-16 06:42:00', format='iso', scale='utc')
end = Time('2010-09-16 06:43:00', format='iso', scale='utc')


# The relevant frame files for those times exist on disk, we just have to find them. The relevant tool for this is the GLUE datafind client. First we open a connection to the default server (set in the environment for the system):


from glue import datafind
connection = datafind.GWDataFindHTTPConnection()


# Then we can ask it to find the frames we care about. These are for the `H` observatory (LIGO Hanford), specifically the `H1_LDAS_C02_L2` data type (representing the LIGO Data Analysis System (LDAS) calibration version 02, at level 2):


framecache = connection.find_frame_urls('H', 'H1_LDAS_C02_L2', start.gps, end.gps, urltype='file')


# Now we know where the data are, we can the strain channel `H1:LDAS-STRAIN` into a `TimeSeries`:


from gwpy.data import TimeSeries
data = TimeSeries.read(framecache, 'H1:LDAS-STRAIN', epoch=start.gps, duration=60)


# Finally, we can drop the data into a plot to see what was going on:


plot = data.plot()
plot.save('hoft.png')


# Out[5]:

# image file:

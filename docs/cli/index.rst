###############################
Command line plotting with GWpy
###############################

LigoDV-web (https://ldvw.ligo.caltech.edu) is a web based tool for viewing LIGO
data.  With the availability of GWpy we have undergone a transformation from
using our own plotting functions to using GWpy.  Since ldvw is written in Java
the most effective way to do this was to develop a command line program to
generate each of the plots.  This program is now part of the GWpy distribution
and can be used on any machine with GWpy installed including Condor compute nodes.


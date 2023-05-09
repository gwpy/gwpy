.. _pydischarge-external-framel:

######
FrameL
######

.. image:: https://img.shields.io/conda/vn/conda-forge/python-framel.svg
   :alt: python-ldas-tools-framecpp conda-forge version
   :target: https://anaconda.org/conda-forge/python-framel
.. image:: https://img.shields.io/conda/pn/conda-forge/python-framel.svg
   :alt: python-ldas-tools-framecpp conda-forge platforms
   :target: https://anaconda.org/conda-forge/python-framel

FrameL is a library for reading and writing data in GWF format defined
in |GWFSpec|_.
The main library is written in C and documented at

https://lappweb.in2p3.fr/virgo/FrameL/

pyDischarge can utilises the Python bindings for FrameL to provide GWF
input/output capabilities.

This optional dependency can be installed using `Conda <https://conda.io>`__
(or `Mamba <https://mamba.readthedocs.io/en/stable/>`__):

.. code-block:: shell
    :name: conda-install-python-framel
    :caption: Install python-framel with conda

    conda install -c conda-forge python-framel

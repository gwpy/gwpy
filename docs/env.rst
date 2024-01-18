.. _gwpy-env:

#####################################
Configuring GWpy from the environment
#####################################

GWpy can be configured by setting environment variables at run time.

.. _gwpy-env-variables:

=====================
Environment variables
=====================

The following variables are understood:

.. list-table:: GWpy environment variables
    :header-rows: 1

    * - Variable
      - Type
      - Default
      - Purpose
    * - ``GWPY_CACHE``
      - :ref:`Bool <gwpy-env-bool>`
      - `False`
      - Whether to cache downloaded files from GWOSC to prevent
        repeated downloads
    * - ``GWPY_FRAME_LIBRARY``
      - String
      - `False`
      - The library to prefer when reading :ref:`GWF <gwpy-timeseries-io-gwf`
        data, one of ``"FrameCPP"``, ``"FrameL"``, ``"LALFrame"``
        (case insensitive)
    * - ``GWPY_RCPARAMS``
      - :ref:`Bool <gwpy-env-bool>`
      - `False`
      - Whether to update `matplotlib.rcParams` with custom GWpy defaults
        for rendering images
    * - ``GWPY_USETEX``
      - :ref:`Bool <gwpy-env-bool>`
      - `False`
      - Whether to use LaTeX when rendering images,
        only used when ``GWPY_RCPARAMS`` is `True`

.. _gwpy-env-bool:

=================
Boolean variables
=================

Many of the variables are boolean switches, meaning they just tell GWpy to
do something, or not to do something. The following values match as `True`:

- ``'y'``
- ``'yes'``
- ``'1'``
- ``'true'``

And these match as `False`:

- ``'n'``
- ``'no'``
- ``'0'``
- ``'false'``

The matching is **case-independent**, so, for example, ``'TRUE'`` will
match as `True`.

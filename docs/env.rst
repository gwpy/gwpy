#####################################
Configuring GWpy from the environment
#####################################

GWpy can be configured by setting environment variables at run time.
Each of the variables are boolean switches, meaning they just tell GWpy to
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

The following variables are defined:

+---------------------+---------+---------------------------------------------+
| Variable            | Default | Purpose                                     |
+=====================+=========+=============================================+
| ``GWPY_CACHE``      | `False` | Whether to cache downloaded files from      |
|                     |         | GWOSC to prevent repeated downloads         |
+---------------------+---------+---------------------------------------------+
| ``GWPY_RCPARAMS``   | `True`  | Whether to update `matplotlib.rcParams`     |
|                     |         | with custom GWpy defaults for rendering     |
|                     |         | images                                      |
+---------------------+---------+---------------------------------------------+
| ``GWPY_USETEX``     | `False` | Whether to use LaTeX when rendering images, |
|                     |         | only used when ``GWPY_RCPARAMS`` is `True`  |
+---------------------+---------+---------------------------------------------+

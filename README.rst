.. role:: bash(code)
   :language: bash

.. role:: python(code)
   :language: python

|Build Status| |Coverage Status| |Documentation Status| |Requirements Status| |PyPI version|

coastlib
========
**coastlib** is a Python library dedicated to solving problems related to the discipline of `coastal engineering <https://en.wikipedia.org/wiki/Coastal_engineering>`_, such as enivronmental data collection (NOAA CO-OPS and NCEI, and WAVEWATCH III, etc.), extreme value analysis (EVA), data analysis and visualization, wave theories, and many more.

**Version:** 1.0.0

**License:** GNU General Public License v3.0

**E-Mail:** bocharovgeorgii@gmail.com

**Documentation:** https://coastlib.readthedocs.io/

Installation
============
.. code:: bash

    pip install coastlib

Jupyter Notebooks
=================
- :python:`coastlib.data` - data extraction tools
  
  - `noaa_coops`_ - NOAA `CO-OPS <https://co-ops.nos.noaa.gov/>`_
  - `noaa_ncei`_ - NOAA `NCEI <https://www.ncei.noaa.gov/>`_

Examples
========
|FentonWave_img|

|rose_plot_1_img| |rose_plot_2_img|

.. |Build Status| image:: https://travis-ci.org/georgebv/coastlib.svg?branch=master
   :target: https://travis-ci.org/georgebv/coastlib
.. |Coverage Status| image:: https://coveralls.io/repos/github/georgebv/coastlib/badge.svg?branch=master
   :target: https://coveralls.io/github/georgebv/coastlib?branch=master
.. |Documentation Status| image:: https://readthedocs.org/projects/coastlib/badge/?version=latest
   :target: https://coastlib.readthedocs.io/en/latest/?badge=latest
.. |Requirements Status| image:: https://requires.io/github/georgebv/coastlib/requirements.svg?branch=master
   :target: https://requires.io/github/georgebv/coastlib/requirements/?branch=master
.. |PyPI version| image:: https://badge.fury.io/py/coastlib.svg
   :target: https://badge.fury.io/py/coastlib

.. _noaa_coops: https://nbviewer.jupyter.org/github/georgebv/coastlib-notebooks/blob/master/notebooks/data/noaa_coops.ipynb
.. _noaa_ncei: https://nbviewer.jupyter.org/github/georgebv/coastlib-notebooks/blob/master/notebooks/data/noaa_ncei.ipynb

.. |FentonWave_img| image:: ./docs/source/example_images/fentonwave.png
.. |rose_plot_1_img| image:: ./docs/source/example_images/rose_plot_1.png
.. |rose_plot_2_img| image:: ./docs/source/example_images/rose_plot_2.png

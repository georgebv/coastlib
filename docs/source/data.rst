Data extraction and processing tools
************************************

Tools related to data extraction and processing are contained within the ``data`` package available through:

.. code:: Python

    from coastlib import data

NOAA CO-OPS Module
==================
The ``noaa_coops`` module is a part of the coastlib.data package. This module provides interface to the NOAA CO-OPS data portal via the CO-OPS API. It allows retrieval of data collected by CO-OPS sensors such as wind, water levels, currents, salinity, air pressure, etc. in the form of pandas DataFrame. With the help of this tool one can automate extraction of large amounts of data from NOAA stations for further processing and storing.

Core tools from this module are available through either of these commands:

>>> from coastlib.data import coops_api, coops_api_batch, coops_datum
>>> from coastlib.data.noaa_coops import coops_api, coops_api_batch, coops_datum

An in-depth tutorial for the ``noaa_coops`` module is available in `this Jupyter notebook <https://nbviewer.jupyter.org/github/georgebv/coastlib-notebooks/blob/master/notebooks/data/noaa_coops.ipynb>`_.

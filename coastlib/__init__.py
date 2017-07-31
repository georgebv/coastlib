#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Georgii Bocharov'
__email__ = 'bocharovgeorgii@gmail.com'
__version__ = '0.1.170729'
__url__ = 'https://github.com/georgebv/coastlib'

# Wave models
from coastlib.wavemodels.airy import AiryWave
from coastlib.wavemodels.fenton import FentonWave

# Plotting tools
from coastlib.core.plotting.rose import rose_plot

# Statistics and analysis tools
from coastlib.core.analyze.extreme import EVA
from coastlib.core.analyze.stats import montecarlo_fit, joint, associated_value

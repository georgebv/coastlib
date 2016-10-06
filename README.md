# coastlib introduction

## Prerequisites
* numpy
* scipy
* pandas
* matplotlib
* statsmodels
* seaborn

## Install
This package is still in early development. Until the proper setup tools are implemented, use the following steps:

* place the folder ***coastlib*** anywhere on your local drive
* edit the system variable **PYTHONPATH** (if it doesn't exist - create it) by adding to its value the full path to the folder containing the ***coastlib*** folder
* import modules, use the help command to get more information about modules and functions

## Package contents

* coreutils
    * adcp_tools
        * `SentinelV` class that is used to extract, convert, store, and export waves and currents data
    * data_analysis_tools    
        * `joint_probability` function that generates joint probability tables
        * `associated_value` function that finds a value statistically associated with another value
    * design_tools
        * `runup`
        * `overtopping`
    * plot_tools
        * `pdf_plots` plots probability density function with a kernel overlay
        * `time_Series_plot` plots time series and finds peaks
        * `rose_plot` plots polar rose plot
        * 'joint_plot` plots bivariate distribution and histrograms on one plot
        * `heatmap` plots heatmap given 2d data (i.e. joint probability table)
* miscutils
    * convenience_tools
        * `ensure_dir` looks for a folder path and if it doesn't exist - creates it and all  parent folders
        * `intersection` finds a list of intersections of horizontal line with a custom profile
        * `splice` function used to splice together multiple dataframes
* models
    * linear_wave_theory
        * `solve_dispersion_relation`
        * `LinearWave` descrbed in the Wiki

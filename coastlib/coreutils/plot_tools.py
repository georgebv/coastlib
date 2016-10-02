import pandas as pd
import scipy.io
import datetime
import detect_peaks
import windrose
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import math
import openpyxl


def pdf_plot(df, **kwargs):
    """
    Plot probability density function (pdf) for parameter *val* (df[val])

    Paramters
    ---------
    df : dataframe
    val : string
        Column name in *df* (i.e. 'Hs')
    savepath : string
        Save folder location
    adcp, month : strings
        Adcp and month values (i.e. 'ADCP1', 'June')
    unit : string
        Unit to display in legend
    """

    val = kwargs.get('val', 'Hs')
    savepath = kwargs.get('savepath', None)
    adcp = kwargs.get('adcp', 'ADCP#')
    month = kwargs.get('month', 'Month')
    unit = kwargs.get('unit', 'ft')

    a = df[pd.notnull(df[val])][val].as_matrix()
    dens = sm.nonparametric.KDEUnivariate(a)
    dens.fit()
    plt.style.use('seaborn-dark')
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.hist(a, bins=50, normed=True, color='blue', rwidth=0.9, label=val+' histogram')
    ax.plot(dens.support, dens.density, lw=2, color='red', label='pdf')
    plt.grid()
    plt.xlabel(val + ' [' + unit + ']')
    plt.ylabel('pdf')
    plt.title(adcp + ' ' + month + ' ' + val + ' pdf')
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath + '\\' + adcp + ' ' + month + ' ' + val + ' pdf.png')
        plt.close()


def time_series_plot(df, col='Hs', **kwargs):
    """
    Plots timeseries of *values* against *time*. Optionally plots peaks over timeseries
    and saves output (figure and peaks table) to a specified location.

    dfparam : pandas DataFrame
        Pandas dataframe with time as index
    col : string
        Column in the pandas dataframe (i.e. 'Hs')
    showpeaks : string (optional)
        'Yes' to find and show peaks, 'No' to not show them
    savepath : string (optional)
        Path to folder which timeseries plot and peaks table are saved to (i.e. 'C:\\foldername').
        Doesn't save files by default (value 'No')
    adcp : string
        Number of the adcp (i.e. 'ADCP1')
    month : string
        Month (i.e. 'June')
    linewidth : float
        Time series line width (default = 1)
    figsize : tuple
        Figure size (default = (16, 8))
    """

    showpeaks = kwargs.get('showpeaks', True)
    savepath = kwargs.get('savepath', 'No')
    adcp = kwargs.get('adcp', 'ADCP#')
    month = kwargs.get('month', 'Month')
    linewidth = kwargs.get('linewidth', 1)
    figsize = kwargs.get('figsize', (16, 8))

    if showpeaks is True:
        indexes = detect_peaks.detect_peaks(df[col].as_matrix())
        x = df[col][indexes].index.values
        y = df[col][indexes].as_matrix()
        if savepath != 'No':
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(df[col], '-', linewidth=linewidth)
            ax.scatter(x, y, s=20, label='Peaks', facecolors='none', edgecolors='r')
            ax.grid()
            plt.xlabel('Time')
            plt.ylabel(col + ' (ft)')
            plt.title(adcp + ' ' + month + ' ' + col + ' time series')
            plt.legend()
            plt.savefig(savepath + '\\' + adcp + ' ' + month + ' ' + col + ' time series.png')
            plt.close()
            writer = pd.ExcelWriter(savepath + '\\' + adcp + ' ' + month + ' ' + col + ' peaks.xlsx')
            df[col][indexes].to_frame().to_excel(writer, sheet_name=col + ' peaks')
            writer.save()
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(df[col], '-', linewidth=linewidth)
            ax.scatter(x, y, s=20, label='Peaks', facecolors='none', edgecolors='r')
            ax.grid()
            plt.xlabel('Time')
            plt.ylabel(col + ' (ft)')
            plt.title(adcp + ' ' + month + ' ' + col + ' time series')
            plt.legend()
    elif savepath != 'No':
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df[col], '-', linewidth=linewidth)
        ax.grid()
        plt.xlabel('Time')
        plt.ylabel(col + ' (ft)')
        plt.title(adcp + ' ' + month + ' ' + col + ' time series')
        plt.legend()
        plt.savefig(savepath + '\\' + adcp + ' ' + month + ' ' + col + ' time series.png')
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df[col], '-', linewidth=linewidth)
        ax.grid()
        plt.xlabel('Time')
        plt.ylabel(col + ' (ft)')
        plt.title(adcp + ' ' + month + ' ' + col + ' time series')
        plt.legend()


def rose_plot(df, val='Hs', direction='Dp',  **kwargs):
    """
    Plots rose of values *val* for directions *direction*
    in dtaframe *df* and saves it to folder *savepath*.

    Parameters
    ----------
    df : pandas dataframe
        Pandas dataframe with columns *val* and *direction*
    val : string
        Column name of values in dataframe *df* (i.e. 'Hs')
    direction : string
        Column name of directions in dataframe *df* (i.e. 'Dp)
    valbins : int
        Number or value bins (default = 6)
    dirbins : int
        Number of directional bins (default = 12)
    savepath : string
        Path to output folder (default = doesn't save) (i.e. 'C:\\folder')
    adcp : string
        Adcp name (used for output naming) (i.e. 'ADCP#')
    month : string
        Month (used for output naming) (i.e. 'June')
    unit : string
        Unit of *val* (i.e. 'ft')
    startfromzero : bool
        Indicates if plot starts from 0 (N) (default = False)
    valbinsize : float
        Set binsize to override *valbins* parameter(default = None, assigns bins automatically)
    """
    valbins = kwargs.get('valbins', 6)
    dirbins = kwargs.get('dirbins', 12)
    savepath = kwargs.get('savepath', 'No')
    adcp = kwargs.get('adcp', 'ADCP#')
    month = kwargs.get('month', 'Month')
    unit = kwargs.get('unit', 'ft')
    startfromzero = kwargs.get('startfromzero', False)
    valbinsize = kwargs.get('valbinsize', None)

    a = df[pd.notnull(df[val])]
    a = a[pd.notnull(a[direction])]
    if startfromzero:
        a[direction] = a[direction].apply(lambda x: x - 0.5 * 360 / dirbins)
    ax = windrose.WindroseAxes.from_ax()
    if valbinsize is not None:
        valbins = np.arange(0, a[val].max()+valbinsize, valbinsize)
    ax.bar(
        a[direction],
        a[val],
        normed=True,
        opening=1,
        edgecolor='black',
        bins=valbins,
        nsector=dirbins,
        cmap=cm.jet,
        startfromzero=startfromzero
    )
    ax.set_legend()
    ax.legend(loc=(-0.1, 0.75), title=val + ' ' + '[' + unit + ']')
    plt.title(adcp + ' ' + month + ' ' + val + ' ' + 'rose', y=1.08)
    if savepath != 'No':
        plt.savefig(savepath + '\\' + adcp + ' ' + month + ' ' + val + ' rose.png')
        plt.close()


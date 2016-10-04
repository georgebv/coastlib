import pandas as pd
import coastlib.thirdpartyutils.detect_peaks as detect_peaks
from coastlib.thirdpartyutils import windrose
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns


def pdf_plot(df, **kwargs):
    """
    Plot probability density function (PDF) for parameter *val* (df[val])

    Paramters
    ---------
    df : dataframe
        Dataframe with column *val*
    val : string
        Column name in *df* (i.e. 'Hs') containing values
    savepath : string
        Save folder location
    xlabel, ylabel, title : string
        X and Y axis labels and plot title
    savename : string
        Name of file
    bins : int
        Number of histogram bins (default = 50)
    plot_style : string
        Plot style (default = 'bmh')
    figsize : tuple
        Figure size (default = (12, 8))
    """
    val = kwargs.get('val', 'Hs')
    savepath = kwargs.get('savepath', None)
    title = kwargs.get('title', 'ADCP# Month Value PDF')
    xlabel = kwargs.get('xlabel', 'Value [ft]')
    ylabel = kwargs.get('ylabel', 'PDF')
    savename = kwargs.get('savename', 'PDF')
    bins = kwargs.get('bins', 50)
    plot_style = kwargs.get('plot_style', 'bmh')
    figsize = kwargs.get('figsize', (12, 8))

    with plt.style.context(plot_style):
        a = df[pd.notnull(df[val])][val].as_matrix()
        dens = sm.nonparametric.KDEUnivariate(a)
        dens.fit()
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(a, bins=bins, normed=True, color='lightskyblue', rwidth=0.9, label='Histogram')
        ax.plot(dens.support, dens.density, lw=2, color='navy', label='PDF')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        if savepath is not None:
            plt.savefig(savepath + '\\' + savename + '.png')
            plt.close()


def time_series_plot(df, **kwargs):
    """
    Plots timeseries of *values* against *time*. Optionally plots peaks over timeseries
    and saves output (figure and peaks table) to a specified location.

    df : pandas DataFrame
        Pandas dataframe with time as index
    val : string
        Column in the pandas dataframe (i.e. 'Hs')
    showpeaks : bool
        Indicates if peaks are found and plotted
    savepath : string (optional)
        Path to folder which timeseries plot and peaks table are saved to (i.e. 'C:\\foldername').
        If not specified, shows plot in a pop-up window.
    savename : string
        Name of file
    linewidth : float
        Time series line width (default = 1)
    figsize : tuple
        Figure size (default = (16, 8))
    xlabel, ylabel, title : string
        X and Y axis labels and plot title
    peaks_outname : string
        Peaks .xlsx output file name
    plot_style : string
        Plot style (default = 'bmh')
    """
    val = kwargs.get('val', 'Hs')
    showpeaks = kwargs.get('showpeaks', True)
    peaks_outname = kwargs.get('peaks_outname', 'Peaks')
    savepath = kwargs.get('savepath', None)
    savename = kwargs.get('savename', 'Time Series')
    linewidth = kwargs.get('linewidth', 0.5)
    figsize = kwargs.get('figsize', (16, 8))
    title = kwargs.get('title', 'Time Series')
    xlabel = kwargs.get('xlabel', 'Time')
    ylabel = kwargs.get('ylabel', 'Value')
    plot_style = kwargs.get('plot_style', 'bmh')
    line_color = kwargs.get('line_color', 'navy')

    with plt.style.context(plot_style):
        if showpeaks:
            indexes = detect_peaks.detect_peaks(df[val].as_matrix())
            x = df[val][indexes].index.values
            y = df[val][indexes].as_matrix()
            if savepath is not None:
                fig, ax = plt.subplots(figsize=figsize)
                ax.plot(df[val], '-', color=line_color, linewidth=linewidth)
                ax.scatter(x, y, s=20, label='Peaks', facecolors='none', edgecolors='r')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                plt.legend()
                plt.savefig(savepath + '\\' + savename + '.png')
                plt.close()
                writer = pd.ExcelWriter(savepath + '\\' + peaks_outname + '.xlsx')
                df[val][indexes].to_frame().to_excel(writer, sheet_name=val + ' peaks')
                writer.save()
            else:
                fig, ax = plt.subplots(figsize=figsize)
                ax.plot(df[val], '-', color=line_color, linewidth=linewidth)
                ax.scatter(x, y, s=20, label='Peaks', facecolors='none', edgecolors='r')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                plt.legend()
        elif savepath is not None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(df[val], '-', color=line_color, linewidth=linewidth)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.savefig(savepath + '\\' + savename + '.png')
            plt.close()
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(df[val], '-', color=line_color, linewidth=linewidth)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()


def rose_plot(df, **kwargs):
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
    savename : string
        Name of file
    startfromzero : bool
        Indicates if plot starts from 0 (N) (default = False)
    valbinsize : float
        Set binsize to override *valbins* parameter (default = None, assigns bins automatically)
    title
    """
    direction = kwargs.get('direction', 'Dp')
    val = kwargs.get('val', 'Hs')
    valbins = kwargs.get('valbins', 6)
    dirbins = kwargs.get('dirbins', 12)
    savepath = kwargs.get('savepath', None)
    savename = kwargs.get('savename', 'Rose')
    title = kwargs.get('title', 'Rose')
    legend = kwargs.get('legend', 'Value [unit]')
    startfromzero = kwargs.get('startfromzero', False)
    valbinsize = kwargs.get('valbinsize', None)

    plt.hist([0, 1])
    plt.close()
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
    ax.legend(loc=(-0.1, 0.75), title=legend, fontsize=9)
    ax.get_legend().get_title().set_fontsize('9')
    plt.title(title, y=1.08, fontsize=16)
    if savepath is not None:
        plt.savefig(savepath + '\\' + savename + '.png')
        plt.close()


def joint_plot(df, **kwargs):
    val1 = kwargs.get('val1', 'Hs')
    val2 = kwargs.get('val2', 'Tp')
    xlabel = kwargs.get('xlabel', 'Hs [ft]')
    ylabel = kwargs.get('ylabel', 'Tp [sec]')
    savepath = kwargs.get('savepath', None)
    savename = kwargs.get('savename', 'Bivariate Distribution')

    with plt.style.context('bmh'):
        g = (sns.JointGrid(x=val1, y=val2, data=df).set_axis_labels(xlabel, ylabel))
        g = g.plot_marginals(sns.distplot, kde=True, color='navy')
        g = g.plot_joint(sns.kdeplot, cmap='plasma')
        g.plot_joint(plt.scatter, c='navy', s=5, linewidth=0.5, marker='x')
    if savepath is not None:
        plt.savefig(savepath + '\\' + savename + '.png')
        plt.close()

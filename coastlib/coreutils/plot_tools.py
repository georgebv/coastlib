import coastlib.thirdpartyutils.detect_peaks as detect_peaks
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.apionly as sns
import statsmodels.api as sm
from scipy import stats
from coastlib.thirdpartyutils import windrose


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
    val = kwargs.pop('val', 'Hs')
    savepath = kwargs.pop('savepath', None)
    title = kwargs.pop('title', 'ADCP# Month Value PDF')
    xlabel = kwargs.pop('xlabel', 'Value [ft]')
    ylabel = kwargs.pop('ylabel', 'PDF')
    savename = kwargs.pop('savename', 'PDF')
    bins = kwargs.pop('bins', 50)
    plot_style = kwargs.pop('plot_style', 'bmh')
    figsize = kwargs.pop('figsize', (12, 8))
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    with plt.style.context(plot_style):
        a = df[pd.notnull(df[val])][val].as_matrix()
        dens = sm.nonparametric.KDEUnivariate(a)
        dens.fit()
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(a, bins=bins, normed=True, color='royalblue', alpha=0.6, rwidth=0.95, label='Histogram')
        ax.plot(dens.support, dens.density, lw=2, color='orangered', label='Kernel PDF')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        if savepath is not None:
            plt.savefig(savepath + '\\' + savename + '.png', bbox_inches='tight', dpi=600)
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
    val = kwargs.pop('val', 'Hs')
    showpeaks = kwargs.pop('showpeaks', True)
    peaks_outname = kwargs.pop('peaks_outname', 'Peaks')
    savepath = kwargs.pop('savepath', None)
    savename = kwargs.pop('savename', 'Time Series')
    linewidth = kwargs.pop('linewidth', 0.5)
    figsize = kwargs.pop('figsize', (16, 8))
    title = kwargs.pop('title', 'Time Series')
    xlabel = kwargs.pop('xlabel', 'Time')
    ylabel = kwargs.pop('ylabel', 'Value')
    plot_style = kwargs.pop('plot_style', 'bmh')
    line_color = kwargs.pop('line_color', 'orangered')
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    with plt.style.context(plot_style):
        if showpeaks:
            indexes = detect_peaks.detect_peaks(df[val].as_matrix())
            x = df[val][indexes].index.values
            y = df[val][indexes].as_matrix()
            if savepath is not None:
                fig, ax = plt.subplots(figsize=figsize)
                ax.plot(df[val], '-', color=line_color, linewidth=linewidth)
                ax.scatter(x, y, s=20, label='Peaks', facecolors='none', edgecolors='royalblue')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                plt.legend()
                plt.savefig(savepath + '\\' + savename + '.png', bbox_inches='tight', dpi=600)
                plt.close()
                writer = pd.ExcelWriter(savepath + '\\' + peaks_outname + '.xlsx')
                df[val][indexes].to_frame().to_excel(writer, sheet_name=val + ' peaks')
                writer.save()
            else:
                fig, ax = plt.subplots(figsize=figsize)
                ax.plot(df[val], '-', color=line_color, linewidth=linewidth)
                ax.scatter(x, y, s=20, label='Peaks', facecolors='none', edgecolors='royalblue')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                plt.legend()
                plt.show()
        elif savepath is not None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(df[val], '-', color=line_color, linewidth=linewidth)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.savefig(savepath + '\\' + savename + '.png', bbox_inches='tight', dpi=600)
            plt.close()
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(df[val], '-', color=line_color, linewidth=linewidth)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.show()


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
    valbin_max : float
        Set the upper limit for value bins (can be useful if bin and color
        consistency is required across multiple roses or to highlight spicific domains)
    title : str
        Plot title
    colormap : module link
        Matplotlib colormap (cm.colormap, i.e. cm.viridis)
    legend : str
        Legend title.
    """
    direction = kwargs.pop('direction', 'Dp')
    val = kwargs.pop('val', 'Hs')
    valbins = kwargs.pop('valbins', 6)
    dirbins = kwargs.pop('dirbins', 12)
    savepath = kwargs.pop('savepath', None)
    savename = kwargs.pop('savename', 'Rose')
    title = kwargs.pop('title', 'Rose')
    legend = kwargs.pop('legend', 'Value [unit]')
    startfromzero = kwargs.pop('startfromzero', False)
    valbinsize = kwargs.pop('valbinsize', None)
    valbin_max = kwargs.pop('valbin_max', None)
    colormap = kwargs.pop('colormap', cm.jet)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    plt.hist([0, 1])
    plt.close()
    a = df[pd.notnull(df[val])]
    a = a[pd.notnull(a[direction])]
    if startfromzero:
        a[direction] = a[direction].apply(lambda x: x - 0.5 * 360 / dirbins)
    ax = windrose.WindroseAxes.from_ax()
    if valbinsize is not None:
        if valbin_max is None:
            valbin_max = a[val].max()
        valbins = np.arange(0, valbin_max + valbinsize, valbinsize)
    ax.bar(
        a[direction],
        a[val],
        normed=True,
        opening=0.95,
        edgecolor=None,
        bins=valbins,
        nsector=dirbins,
        cmap=colormap,
        startfromzero=startfromzero
    )
    ax.set_legend()
    ax.legend(loc=(-0.12, 0.75), title=legend, fontsize=9)
    ax.get_legend().get_title().set_fontsize('9')
    ax.grid('on', linestyle=':')
    plt.title(title, y=1.08, fontsize=16)
    if savepath is not None:
        plt.savefig(savepath + '\\' + savename + '.png', bbox_inches='tight', dpi=600)
        plt.close()


def joint_plot(df, **kwargs):
    """
    Plots bivariate distribution.

    Parameters
    ----------
    df : dataframe
        Pandas dataframe
    val1 : string
        Value 1 (i.e. 'Hs')
    val2 : string
        Value 2 (i.e. 'Tp')
    xlabel, ylabel : string
        Axes labels
    savepath, savename : string
        Path to save folder and file name
    figsize : float
        Figure size ( will be a square)
    """
    val1 = kwargs.pop('val1', 'Hs')
    val2 = kwargs.pop('val2', 'Tp')
    xlabel = kwargs.pop('xlabel', 'Hs [ft]')
    ylabel = kwargs.pop('ylabel', 'Tp [sec]')
    savepath = kwargs.pop('savepath', None)
    savename = kwargs.pop('savename', 'Bivariate Distribution')
    figsize = kwargs.pop('figsize', 10)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    with plt.style.context('bmh'):
        g = (sns.JointGrid(x=val1, y=val2, data=df, size=figsize).set_axis_labels(xlabel, ylabel))
        g = g.plot_marginals(sns.distplot, kde=True, color='navy')
        g = g.plot_joint(sns.kdeplot, cmap='plasma')
        g = g.plot_joint(plt.scatter, c='navy', s=5, linewidth=0.5, marker='x')
        g.annotate(
            (lambda a, b: stats.pearsonr(a, b)[0] ** 2),
            template='{stat} = {val:.2f}',
            stat='$R^2$',
            fontsize=12,
            loc='upper right'
        )
    if savepath is not None:
        plt.savefig(savepath + '\\' + savename + '.png', bbox_inches='tight', dpi=600)
        plt.close()


def heatmap(df, **kwargs):
    """
    Plots heatmap with cell values displayed.

    df : dataframe

    figsize : tuple
        Figure size (optional)
    xlabel, ylabel, title : string
        X and Y axis labels and plot title
    savepath : string (optional)
        Path to folder which timeseries plot and peaks table are saved to (i.e. 'C:\\foldername').
        If not specified, shows plot in a pop-up window.
    savename : string
        Name of file
    yaxflip : bool
        Flip y axis (default = True)
    """
    title = kwargs.pop('title', 'Joint probability')
    xlabel = kwargs.pop('xlabel', 'Hs [ft]')
    ylabel = kwargs.pop('ylabel', 'Tp [sec]')
    savepath = kwargs.pop('savepath', None)
    savename = kwargs.pop('savename', 'Heatmap')
    figsize = kwargs.pop('figsize', (1.5 * len(df.columns), 1.2 * len(df)))
    yaxflip = kwargs.pop('yaxflip', True)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    plt.figure(figsize=figsize)
    ax = sns.heatmap(df, annot=True, linewidths=.5, fmt='.2f', square=True)
    plt.yticks(rotation=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel, rotation=0)
    plt.title(title, y=1.04)
    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.04, 1.04)
    if yaxflip:
        ax.invert_yaxis()
    if savepath is not None:
        plt.savefig(savepath + '\\' + savename + '.png', bbox_inches='tight', dpi=600)
        plt.close()

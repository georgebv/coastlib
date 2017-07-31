import matplotlib.pyplot as plt
import pandas as pd
import seaborn.apionly as sns
import statsmodels.api as sm
from scipy import stats


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

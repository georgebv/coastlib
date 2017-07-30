import math
import pandas as pd
import statsmodels.api as sm
import numpy as np
import datetime
import scipy.stats as sps
import scipy.interpolate
from scipy.optimize import curve_fit
import scipy.optimize
from scipy import stats
import matplotlib.pyplot as plt
import warnings


def joint_probability(df, val1='Hs', val2='Tp', binsize1=0.3, binsize2=4, **kwargs):
    """
    Generates a joint probability table of 2 variables. (Works only for positive values!)

    Parameters
    ----------
    df : dataframe
        Pandas dataframe
    val1, val2 : str
        Column names in df
    binsize1, binsize2 : float
        Bin sizes for variables val1 and val2
    savepath, savename : str
        Save folder path and file save name
    output_format : str
        Joint table values (absolute 'abs' or relative / percent 'rel')
    """
    savepath = kwargs.pop('savepath', None)
    savename = kwargs.pop('savename', 'Joint Probability')
    output_format = kwargs.pop('output_format', 'rel')
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    a = df[pd.notnull(df[val1])]
    a = a[pd.notnull(a[val2])]
    if a[val1].min() < 0:
        shift1 = - (a[val1].min() - a[val1].min() % (- binsize1)) + binsize1
    else:
        shift1 = 0
    if a[val2].min() < 0:
        shift2 = - (a[val2].min() - a[val2].min() % (- binsize2)) + binsize2
    else:
        shift2 = 0
    a[val1] = a[val1].values + shift1
    a[val2] = a[val2].values + shift2
    bins1 = math.ceil(a[val1].max() / binsize1) - 1
    bins2 = math.ceil(a[val2].max() / binsize2) - 1
    columns = ['(-inf ; {0:.1f})'.format(- shift1 + binsize1)]
    rows = ['(-inf ; {0:.1f})'.format(- shift2 + binsize2)]
    for i in range(1, bins1):
        low = i * binsize1
        up = low + binsize1
        columns += ['[{0:.1f} ; {1:.1f})'.format(low - shift1, up - shift1)]
    columns += ['[{0:.1f} ; inf)'.format(bins1 * binsize1 - shift1)]
    for i in range(1, bins2):
        low = i * binsize2
        up = low + binsize2
        rows += ['[{0:.1f} ; {1:.1f})'.format(low - shift2, up - shift2)]
    rows += ['[{0:.1f} ; inf)'.format(bins2 * binsize2 - shift2)]
    if output_format == 'abs':
        jp_raw = pd.DataFrame(0, index=rows, columns=columns)
    else:
        jp_raw = pd.DataFrame(.0, index=rows, columns=columns)

    tot = len(a)
    for i in range(bins2 + 1):
        bin2_low = i * binsize2
        bin2_up = bin2_low + binsize2
        if i == bins2:
            bin2_up = np.inf
        if i == 0:
            bin2_low = -np.inf
        for j in range(bins1 + 1):
            bin1_low = j * binsize1
            bin1_up = bin1_low + binsize1
            if j == bins1:
                bin1_up = np.inf
            if j == 0:
                bin1_low = -np.inf
            b = len(
                a[
                    (a[val1] < bin1_up) &
                    (a[val1] >= bin1_low) &
                    (a[val2] < bin2_up) &
                    (a[val2] >= bin2_low)
                ]
            )
            if output_format == 'abs':
                jp_raw[columns[j]][i] = b
            elif output_format == 'rel':
                jp_raw[columns[j]][i] = b / tot
            else:
                raise ValueError('output format should be either *abs* or *rel*')
    if savepath is not None:
        jp_raw.to_excel(pd.ExcelWriter(savepath + '\\' + savename + '.xlsx'), sheet_name='joint_prob', )
    else:
        return jp_raw


def associated_value(df, value, search_range, val1='Hs', val2='Tp', confidence=0.5, plot_cdf=False):
    """
    Calculates a statistically associated value for a series of 2 correllated values (joint probability)

    Parameters
    ----------
    df : dataframe
        Pandas dataframe
    val1, val2 : str
        Column names in dataframe df
    value : float
        Value of val1 for which an associated value val2 is found
    search_range : float
        Range of val1 within which values of val2 will be extraced for analysis (half bin size from
        joint probability)
    confidence : float
        Confidence for associated value - shows probability of non-exceedance (default 0.5 - median value)
    plot_cdf : bool
        If True - display a CDF plot of val2 in range val1 ± search_range
    """

    df = df[pd.notnull(df[val1])]
    df = df[pd.notnull(df[val2])]
    target = df[(df[val1] >= value - search_range) & (df[val1] <= value + search_range)]
    kde = sm.nonparametric.KDEUnivariate(target[val2].values)
    kde.fit()
    fit = scipy.interpolate.interp1d(kde.cdf, kde.support, kind='linear')
    if plot_cdf == True:
        with plt.style.context('bmh'):
            plt.plot(kde.support, kde.cdf, lw=1, color='orangered')
            plt.title('CDF of {0} for {1} in range [{low} - {top}]'.
                      format(val2, val1, low=round(value - search_range, 2), top=round(value + search_range, 2)))
            plt.ylabel('CDF')
            plt.xlabel(val2)
            plt.annotate(r'{perc}% Associated value {0}={1}'.format(val2, round(fit(confidence).tolist(), 2),
                                                                   perc=confidence*100),
                         xy=(fit(confidence).tolist(), confidence),
                         xytext=(fit(0.5).tolist()+search_range, 0.5),
                         arrowprops=dict(facecolor='k', shrink=0.01))
    return fit(confidence).tolist()


def montecarlo_fit(function, bounds, x, y, x_new, confidence=90, sims=1000, **kwargs):
    '''
    Fits function <function> to the <x,y> locus of points and evaluates the fit for a new set of values <x_new>.
    Returns <lower ci, fit, upper ci> using <how> method for a confidence interval <confidence>.

    Parameters
    ----------
    function
    bounds
    x
    y
    x_new
    confidence
    sims
    sample

    Returns
    -------

    '''
    sample = kwargs.pop('sample', 0.4)
    poisson = kwargs.pop('poisson', True)
    how = kwargs.pop('how', 'kde')
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    popt, pcov = curve_fit(function, x, y, bounds=bounds)
    popt_s = []
    if how == 'montecarlo':
        for i in range(sims):
            idx = np.random.choice(np.arange(len(x)), int(len(x) * sample), replace=False)
            x_s, y_s = x[idx], y[idx]
            popt_l, pcov_l = curve_fit(function, x_s, y_s, bounds=bounds)
            popt_s += [popt_l]
    elif how == 'kde':
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values)
        for i in range(sims):
            if poisson:
                # Generate a sample of a size <len_sample> from Poisson distibution and fit function to it
                len_sample = sps.poisson.rvs(len(x))
                sample = kernel.resample(len_sample)
            else:
                sample = kernel.resample(len(x))
            x_s, y_s = sample[0], sample[1]
            popt_l, pcov_l = curve_fit(function, x_s, y_s, bounds=bounds)
            popt_s += [popt_l]
    else:
        raise ValueError('ERROR: Method {} not recognized.'.format(how))

    y_new = function(x_new, *popt)
    y_s = [function(x_new, *j) for j in popt_s]
    y_s_pivot = np.array([[y_s[i][j] for i in range(len(y_s))] for j in range(len(x_new))])
    moments = [sps.norm.fit(x) for x in y_s_pivot]
    intervals = [sps.norm.interval(alpha=confidence/100, loc=x[0], scale=x[1]) for x in moments]
    lower = [x[0] for x in intervals]
    upper = [x[1] for x in intervals]
    return lower, y_new, upper

import pandas as pd
import statsmodels.api as sm
import numpy as np
import scipy.stats as sps
import scipy.interpolate
from scipy.optimize import curve_fit
import scipy.optimize
from scipy import stats
import matplotlib.pyplot as plt


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

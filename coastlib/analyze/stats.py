import numpy as np
import pandas as pd
import warnings
import scipy.stats
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt
import statsmodels.nonparametric.kde


def joint(value_1, value_2, binsize_1=0.3, binsize_2=4, relative=False):
    """
    Generates a joint probability table of 2 variables
    Filtering data before and removing empty columns after is up to user

    Mandatory input
    ===============
    value_1, value_2 : 1D lists or arrays
        Arrays of equal length

    Optional input
    ==============
    binsize_1, binsize_2 : float (default=0.3,4)
        Bin sizes for variables value_1 and value_2
    relative : bool (default=False)
        If True, returns relative probabilities (<=1)

    Output
    ======
    Pandas dataframe with a joint probability table
    """

    if (not isinstance(value_1, np.ndarray)) or (not isinstance(value_2, np.ndarray)):
        try:
            value_1 = np.array(value_1)
            value_2 = np.array(value_2)
        except Exception as _e:
            raise ValueError('{}\n'
                             'Input values should be 1D lists or arrays.'.format(_e))

    data = pd.DataFrame(data=value_1, columns=['v1'])
    data['v2'] = value_2

    def _round(_x):
        return float(format(_x, '.5f'))

    _b1min = _round(value_1.min() - value_1.min() % binsize_1)
    _b1max = _round(value_1.max() - value_1.max() % binsize_1 + binsize_1)

    _b2min = _round(value_2.min() - value_2.min() % binsize_2)
    _b2max = _round(value_2.max() - value_2.max() % binsize_2 + binsize_2)

    bots_1 = np.arange(_b1min-binsize_1, _b1max+binsize_1, binsize_1)
    bots_2 = np.arange(_b2min-binsize_2, _b2max+binsize_2, binsize_2)

    index_1 = ['(-inf ; {0:.2f}]'.format(bots_1[1])]
    for bot in bots_1[1:-1]:
        index_1.extend(['[{0:.2f} ; {1:.2f})'.format(bot, bot + binsize_1)])
    index_1.extend(['[{0:.2f} ; inf)'.format(bots_1[-1])])

    index_2 = ['(-inf ; {0:.2f}]'.format(bots_2[1])]
    for bot in bots_2[1:-1]:
        index_2.extend(['[{0:.2f} ; {1:.2f})'.format(bot, bot + binsize_2)])
    index_2.extend(['[{0:.2f} ; inf)'.format(bots_2[-1])])

    bins = [[_round(bot), _round(bot + binsize_1)] for bot in bots_1]
    datas = [data[(data['v1'] >= bin[0]) & (data['v1'] < bin[1])] for bin in bins]

    table = np.zeros(shape=(len(index_1), len(index_2)))
    for i, _data in enumerate(datas):
        for j, bot_2 in enumerate(bots_2):
            top_2 = bot_2 + binsize_2
            table[i][j] = (
                (_data['v2'] >= bot_2) &
                (_data['v2'] < top_2)
            ).sum()

    if not np.isclose(len(value_1), table.sum()):
        warnings.warn('THE RESULT IS WRONG. Missing {} values.'.format(len(value_1) - table.sum()))

    if relative:
        table /= len(data)

    return pd.DataFrame(data=table, index=index_1, columns=index_2)


def montecarlo_fit(function, x, y, x_new, confidence=90, sims=1000, **kwargs):
    '''
    Fits function <function> to the <x,y> locus of points and evaluates the fit for a new set of values <x_new>.
    Returns <lower ci, fit, upper ci> using <how> method for a confidence interval <confidence>.

    Mandatory inputs
    ================
    function : callable
    x : list or array
    y : list or array
    x_new : list or array

    Optional inputs
    ===============
    confidence : float (default=90)
    sims : int (default=1000)
    sample : float (default=0.4)
    poisson : bool (default=True)
    how : str (default='kde')
        'montecarlo', 'kde'
    bounds : 2-tuple of array-like (default=(-np.inf, np.inf) - equivalent to no bounds)
        Bounds in the form of (lower, upper) where lower and upper are lists of bounds for each of the
        independent variables in <function>. If float, then all variables have this bound.

    Output
    ======
    (y_new, lower, upper) : tuple of <numpy.ndarray>s
        a tuple with (y_new fitted to <x_new>, lower bound for <confidence>, upper bound for <confidence>)
    '''

    sample = kwargs.pop('sample', 0.4)
    poisson = kwargs.pop('poisson', True)
    how = kwargs.pop('how', 'kde')
    bounds = kwargs.pop('bounds', (-np.inf, np.inf))
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    popt, pcov = scipy.optimize.curve_fit(function, x, y, bounds=bounds)
    popt_s = []
    if how == 'montecarlo':
        for i in range(sims):
            idx = np.random.choice(np.arange(len(x)), int(len(x) * sample), replace=False)
            x_s, y_s = x[idx], y[idx]
            popt_l, pcov_l = scipy.optimize.curve_fit(function, x_s, y_s, bounds=bounds)
            popt_s.extend([popt_l])
    elif how == 'kde':
        values = np.vstack([x, y])
        kernel = scipy.stats.gaussian_kde(values)
        for i in range(sims):
            if poisson:
                # Generate a sample of a size <len_sample> from Poisson distibution and fit function to it
                len_sample = scipy.stats.poisson.rvs(len(x))
                kde_sample = kernel.resample(len_sample)
            else:
                kde_sample = kernel.resample(len(x))
            x_s, y_s = kde_sample[0], kde_sample[1]
            popt_l, pcov_l = scipy.optimize.curve_fit(function, x_s, y_s, bounds=bounds)
            popt_s.extend([popt_l])
    else:
        raise ValueError('Method {} not recognized.'.format(how))

    y_new = function(x_new, *popt)
    y_s = np.array([function(x_new, *j) for j in popt_s])
    moments = [scipy.stats.norm.fit(x) for x in y_s.T]
    intervals = [scipy.stats.norm.interval(alpha=confidence/100, loc=x[0], scale=x[1]) for x in moments]
    lower = [x[0] for x in intervals]
    upper = [x[1] for x in intervals]
    return np.array(y_new), np.array(lower), np.array(upper)


def associated_value(values_1, values_2, value, search_range, confidence=0.5, plot_cdf=False):
    """
    Calculates a statistically associated value for a series of 2 correllated values (joint probability)

    Mandatory inputs
    ================
    values_1, values_2 : array
        Arrays with
    value : float
        Value of <values_1> for which an associated value from <values_2> is found
    search_range : float
        Range of <values_1> within which values of <values_2> will be extraced for analysis
        (searches within ± <search_range> of <value>)

    Optional inputs
    ===============
    confidence : float (default=0.5)
        Confidence for associated value - shows probability of non-exceedance (default 0.5 - median value)
    plot_cdf : bool (default=False)
        If True - display a CDF plot of <values_2> in range <value> ± <search_range>

    Output
    ======
     : float
        a value from <values_2> associated with <value> from <values_1>
    """

    df = pd.DataFrame(data=values_1, columns=['v1'])
    df['v2'] = values_2

    target = df[(df['v1'] >= value - search_range) & (df['v1'] <= value + search_range)]
    kde = statsmodels.nonparametric.kde.KDEUnivariate(target['v2'].values)
    kde.fit()
    fit = scipy.interpolate.interp1d(kde.cdf, kde.support, kind='slinear')
    if plot_cdf:
        with plt.style.context('bmh'):
            plt.plot(kde.support, kde.cdf, lw=1, color='orangered')
            plt.title('CDF of value_1 for value_2 in range [{low} - {top}]'.
                      format(low=round(value - search_range, 2), top=round(value + search_range, 2)))
            plt.ylabel('CDF')
            plt.xlabel('value_2')
            plt.annotate(
                r'{perc}% Associated value {0}={1}'.format('value_1', np.round(fit(confidence).tolist(), 2),
                                                           perc=confidence * 100),
                xy=(float(fit(confidence)), confidence),
                xytext=(float(fit(0.5)) + search_range, 0.5),
                arrowprops=dict(facecolor='k', shrink=0.01)
            )
    return float(fit(confidence))

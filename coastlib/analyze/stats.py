import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.optimize
import scipy.stats
import statsmodels.nonparametric.kde


def joint_table(values_1, values_2, binsize_1=0.3, binsize_2=4, bins=None, relative=False):
    """
    Generates a joint probability table of 2 variables
    Filtering data before and removing empty columns after is up to user

    Mandatory inputs
    ================
    value_1, value_2 : 1D lists or arrays
        Arrays of equal length

    Optional inputs
    ===============
    binsize_1, binsize_2 : float (default=0.3,4)
        Bin sizes for variables value_1 and value_2
    bins : tuple of arrays (default=None) (bins_1, bins_2)
        Boundaries of bins for <values_1> and <values_2>. If <bins> are provided, binsizes are ignored
    relative : bool (default=False)
        If True, returns relative probabilities (<=1)

    Output
    ======
    Pandas dataframe with a joint probability table
    """

    if (not isinstance(values_1, np.ndarray)) or (not isinstance(values_2, np.ndarray)):
        try:
            values_1 = np.array(values_1)
            values_2 = np.array(values_2)
        except Exception as _e:
            raise ValueError('{}\n'
                             'Input values should be 1D lists or arrays.'.format(_e))
    if not bins:
        if binsize_1 <=0 or binsize_2 <= 0:
            raise ValueError('Bin sizes must be positive numbers')

        if values_1.min() >= 0:
            _b1min = values_1.min() - np.abs(values_1.min() % binsize_1)
        else:
            _b1min = values_1.min() + (binsize_1 - np.abs(values_1.min() % binsize_1)) - binsize_1
        if values_1.max() >= 0:
            _b1max = values_1.max() - np.abs(values_1.max() % binsize_1) + binsize_1
        else:
            _b1max = values_1.max() + (binsize_1 - np.abs(values_1.max() % binsize_1))

        if values_2.min() >= 0:
            _b2min = values_2.min() - np.abs(values_2.min() % binsize_2)
        else:
            _b2min = values_2.min() + (binsize_2 - np.abs(values_2.min() % binsize_2)) - binsize_2
        if values_2.max() >= 0:
            _b2max = values_2.max() - np.abs(values_2.max() % binsize_2) + binsize_2
        else:
            _b2max = values_2.max() + (binsize_2 - np.abs(values_2.max() % binsize_2))

        bots_1 = np.arange(_b1min, _b1max + binsize_1, binsize_1) * 1.0
        bots_2 = np.arange(_b2min, _b2max + binsize_2, binsize_2) * 1.0

        index_1 = []
        for bot in bots_1[0:-1]:
            index_1.extend(['[{0:.2f} ; {1:.2f})'.format(bot, bot + binsize_1)])

        index_2 = []
        for bot in bots_2[0:-1]:
            index_2.extend(['[{0:.2f} ; {1:.2f})'.format(bot, bot + binsize_2)])

        bots_1[0], bots_1[-1] = -np.inf, np.inf
        bots_2 = 1.0 * bots_2
        bots_2[0], bots_2[-1] = -np.inf, np.inf
    else:
        bots_1, bots_2 = bins[0], bins[1]

        index_1 = []
        for bot in bots_1[0:-1]:
            index_1.extend(['[{0:.2f} ; {1:.2f})'.format(bot, bot + binsize_1)])

        index_2 = []
        for bot in bots_2[0:-1]:
            index_2.extend(['[{0:.2f} ; {1:.2f})'.format(bot, bot + binsize_2)])

    table = np.histogram2d(values_1, values_2, [bots_1, bots_2], normed=False)

    if not np.isclose(len(values_1), table[0].sum()):
        warnings.warn('THE RESULT IS WRONG. Missing {} values.'.format(len(values_1) - table[0].sum()))

    if relative:
        return pd.DataFrame(data=table[0] / len(values_1), index=index_1, columns=index_2)
    else:
        return pd.DataFrame(data=table[0], index=index_1, columns=index_2)



def confidence_fit(function, x, y, x_new, confidence=95, sims=1000, **kwargs):
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
    confidence : float (default=95)
    sims : int (default=1000)
    resample_ratio : float (default=0.4)
    poisson : bool (default=True)
    how : str (default='kde')
        'resample', 'kde'
    bounds : 2-tuple of array-like (default=(-np.inf, np.inf) - equivalent to no bounds)
        Bounds in the form of (lower, upper) where lower and upper are lists of bounds for each of the
        independent variables in <function>. If float, then all variables have this bound.

    Output
    ======
    (y_new, lower, upper) : tuple of <numpy.ndarray>s
        a tuple with (y_new fitted to <x_new>, lower bound for <confidence>, upper bound for <confidence>)
    '''

    resample_ratio = kwargs.pop('resample_ratio', 0.4)
    poisson = kwargs.pop('poisson', True)
    how = kwargs.pop('how', 'kde')
    bounds = kwargs.pop('bounds', (-np.inf, np.inf))
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    # Initial fit
    popt, pcov = scipy.optimize.curve_fit(function, x, y, bounds=bounds)

    # Confidence interval
    popt_s = []
    if how == 'resample':
        for i in range(sims):
            idx = np.random.choice(np.arange(len(x)), int(len(x) * resample_ratio), replace=False)
            x_s, y_s = x[idx], y[idx]
            popt_l, pcov_l = scipy.optimize.curve_fit(function, x_s, y_s, bounds=bounds)
            popt_s.append(popt_l)
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
            popt_s.append(popt_l)
    else:
        raise ValueError('Method {} not recognized.'.format(how))

    y_new = function(x_new, *popt)
    y_s = np.array([function(x_new, *j) for j in popt_s])
    moments = [scipy.stats.norm.fit(x) for x in y_s.T]
    intervals = [scipy.stats.norm.interval(alpha=confidence / 100, loc=x[0], scale=x[1]) for x in moments]
    lower = [x[0] for x in intervals]
    upper = [x[1] for x in intervals]
    return np.array(y_new), np.array(lower), np.array(upper)


def associated_value(values_1, values_2, value, search_range, plot_cdf=False):
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

    mask = (values_1 >= (value - search_range)) & (values_1 <= (value + search_range))
    target = values_2[mask]

    pdf_hist = np.histogram(target, normed=True)
    empirical_support = [(pdf_hist[1][i] + pdf_hist[1][i+1])/2 for i in range(len(pdf_hist[1])-1)]
    empirical_pdf = pdf_hist[0]
    empirical_cdf = np.cumsum(np.histogram(target)[0] / len(target))

    kernel = statsmodels.nonparametric.kde.KDEUnivariate(target.astype(np.float))
    kernel.fit()
    support = np.linspace(empirical_support[0], empirical_support[-1], 100)
    mask = (kernel.support >= support[0]) & (kernel.support <= support[-1])
    kernel_support = kernel.support[mask]
    kernel_cdf = kernel.cdf[mask]

    with plt.style.context('bmh'):
        plt.hist(target, normed=True, alpha=0.6, rwidth=0.9, color='royalblue')
        plt.plot(support, kernel.evaluate(support), color='orangered', lw=2)
        plt.plot(empirical_support, empirical_pdf, color='k', lw=2, ls='--')

    with plt.style.context('bmh'):
        plt.hist(target, normed=True, alpha=0.6, rwidth=0.9, color='royalblue', cumulative=True)
        plt.plot(kernel_support, kernel_cdf, color='orangered', lw=2)
        plt.plot(empirical_support, empirical_cdf, drawstyle='steps-post', color='k', lw=2, ls='--')

    print('Associated value ->', kernel_support[kernel.evaluate(kernel_support).argmax()])

    # if plot_cdf:
    #     with plt.style.context('bmh'):
    #         plt.plot(kde.support, kde.cdf, lw=1, color='orangered')
    #         plt.title('CDF of value_1 for value_2 in range [{low} - {top}]'.
    #                   format(low=round(value - search_range, 2), top=round(value + search_range, 2)))
    #         plt.ylabel('CDF')
    #         plt.xlabel('value_2')
    #         plt.annotate(
    #             r'{perc}% Associated value {0}={1}'.format('value_1', np.round(fit(confidence).tolist(), 2),
    #                                                        perc=confidence * 100),
    #             xy=(float(fit(confidence)), confidence),
    #             xytext=(float(fit(0.5)) + search_range, 0.5),
    #             arrowprops=dict(facecolor='k', shrink=0.01)
    #         )
    return kernel_support[kernel.evaluate(kernel_support).argmax()]

import datetime

import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


class EVA:
    """
    Extreme Value Analysis class. Takes a Pandas DataFrame with values in <column> and with datetime index.
    Extracts extreme values. Assists with threshold value selection. Fits data to distributions via MLE.
    Returns extreme values' return periods and estimates confidence intervals. Generates data plots.

    Typical Workflow
    ================
    1   Create a class instance by passing a Pandas Dataframe to the <dataframe> parameter and
        column with values to <column>. Set <discontinuous> to True, if the dataset has gaps -
        aka years without any data.
        ~$ eve = EVA(dataframe=df, column=col, discontinuous=True)

    2   Use the <.get_extremes> class method to parse extreme values. Set <method> to 'AM' to use
        annual maxima and to 'POT' to use peaks over threshold extraction methods.
        ~$ eve.get_extremes(method='POT', **kwargs)

    3   Use the <.fit> class method to fit a distribution <distribution> to the extracted extreme values.
        The parameter <distribution> is a string with a scipy.stats distribution name
        (look up on https://docs.scipy.org/doc/scipy/reference/stats.html).
        ~$ eve.fit(distribution='genpareto', confidence_interval=0.95, **kwargs)

        Recommended distributions: expon, exponweib (aka classic Weibull), frechet_l (or _r),
        genpareto, genextreme, genexpon, gumbel_r, invgauss, invweibull, lognorm, powerlognorm,
        pearson3, pareto, rayleigh, weibull_min (or _max)

    4   Use the <.plot> method to get a quick visual summary - shows observed pdf, cdf, extreme values
        vs. the distribution-predicted smooth fit

    """

    def __init__(self, dataframe, column=None, discontinuous=True):
        # Ensure passed data is a Pandas Dataframe object (pd.Dataframe)
        if not isinstance(dataframe, pd.DataFrame):
            try:
                self.data = dataframe.to_frame()
            except AttributeError:
                raise TypeError('Invalid data type in <df>.'
                                ' EVA takes only Pandas DataFrame or Series objects.')
        else:
            self.data = dataframe

        # Check passed column value is valid
        if column:
            if column in self.data.columns:
                self.column = column
            else:
                raise ValueError('Column {0} cannot be accessed. Check spelling'.format(column))
        else:
            self.column = self.data.columns[0]

        # Verify all years are accounted for (even those without data - i.e. gaps in time series)
        if discontinuous:
            self.N = len(np.arange(self.data.index.year.min(), self.data.index.year.max()+1, 1))
        else:
            self.N = len(np.unique(self.data.index.year))

        if self.N != len(np.arange(self.data.index.year.min(), self.data.index.year.max()+1, 1)):
            missing = []
            for _year in np.arange(self.data.index.year.min(), self.data.index.year.max()+1, 1):
                if _year not in np.unique(self.data.index.year):
                    missing.append(_year)
            print(
                '\n\nData is not continuous!\nMissing years {0}\n'
                'Set <dicontinuous=True> to account for all years, '
                'assuming there were NO peaks in the missing years.\n'
                'Without this option turned on, extreme events might be'
                'significantly overestimated (conservative results)'
                'due to the total length of observation period being low'.format(missing)
            )

    def get_extremes(self, method='POT', **kwargs):
        """
        Extracts extreme values from provided data using POT (peaks over threshold)
        or AM (annual maxima) methods. If method is POT, also declusters extreme values using
        the run method (aka minimum distance between events).

        Parameters
        ----------
        method : str
            Peak extraction method. POT for peaks over threshold and AM for annual maxima.

        kwargs
            decluster : bool
                POT method only: decluster checks if extremes are declustered (default=True)
            threshold : float
                POT method only: threshold for extreme value extraction
            r : float
                POT method only: minimum distance in hours between events for them to be considered independent
                (default=24)

        Returns
        -------
        Creates a self.extremes dataframe with extreme values and return periods determined using
        the Weibull plotting position P=m/(N+1)
        """

        self.method = method

        if self.method == 'POT':
            # Parse optional POT-only arguments
            decluster = kwargs.pop('decluster', True)
            self.threshold = kwargs.pop('threshold')
            r = datetime.timedelta(hours=kwargs.pop('r', 24))
            assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

            # Extract raw extremes
            self.extremes = self.data[self.data[self.column] > self.threshold]

            # Decluster raw extremes
            if decluster:
                new_indexes, new_values = [self.extremes.index[0]], [self.extremes[self.column][0]]
                for _index, _value in zip(self.extremes.index, self.extremes[self.column].values):
                    if _index - new_indexes[-1] > r:
                        new_indexes.append(_index)
                        new_values.append(_value)
                    else:
                        new_indexes[-1] = _index
                        if _value > new_values[-1]:
                            new_values[-1] = _value
                self.extremes = pd.DataFrame(data=new_values, index=new_indexes, columns=[self.column])
        elif self.method == 'AM':
            self.threshold = 0
            # Extract Annual Maxima extremes
            extreme_indexes, extreme_values = [], []
            for _year in np.unique(self.data.index.year):
                extreme_values.append(
                    self.data[self.data.index.year == _year][self.column].values.max()
                )
                # If several equal maximum values exist throughout a year, the first one is taken
                extreme_indexes.append(
                    self.data[
                        (self.data[self.column].values == extreme_values[-1]) & (self.data.index.year == _year)
                    ].index[0]
                )
            self.extremes = pd.DataFrame(data=extreme_values, index=extreme_indexes, columns=[self.column])

        # Estimate return periods for extracted extreme values using the Weibull plotting position
        self.extremes.sort_values(by=self.column, ascending=True, inplace=True)
        self.rate = len(self.extremes) / self.N
        if self.method == 'AM' and self.rate != 1:
            raise ValueError(
                'For the Annual Maxima method the number of extreme events ({0}) should be '
                'equal to number of years ({1}).\n'
                'Consider using <discontinuous=False> when declaring the EVA object.'.format(len(self.extremes), self.N)
            )
        cdf = np.arange(1, len(self.extremes) + 1) / (len(self.extremes) + 1)
        sf = 1 - cdf  # survival function, aka annual probability of exceedance
        self.extremes['T'] = 1 / sf / self.rate
        self.extremes.sort_index(inplace=True)

        # Remove previously fit distribution to avoid mistakes
        try:
            del self.retvalsum
        except AttributeError:
            pass

    def fit(self, distribution='genpareto', confidence_interval=None, **kwargs):
        """
        Fits <distribution> to extracted extreme values. Creates <retvalsum> dataframe with
        estimated extreme values for return periods and upper and lower confidence bounds, if specified.

        Parameters
        ----------
        distribution : str
            Scipy disttribution name (default='genpareto')
        confidence_interval : float
            Confidence interval width, should be confidence_interval<1 or None. Not estimated if None (default=None)
        kwargs : dict
            loc : float
                Guess of the location parameter value. Location parameter is estimated if None is passed (default=None)
            confidence_method : str
                Confidence interval estimation methods (default='bootstrap'):
                    -'montecarlo' - montecarlo method. Generates poisson-distributed sized samples
                        from the fitted distribution
                    -'jackknife' - jackknife method. Generates leave-one-out samples
                        from the original extracted extreme values TODO - not yet implemented
                    -'bootstrap' - bootstrap method. Generates poisson-distributed sized samples
                        from the original extracted extreme values with replacement
            k : int
                Number of samples used to estimate confidence bounds, !!!highly affect performance!!! (default=10**2)
            truncate : bool
                Set True to remove samples producing extreme outliers - defined as being higher than return values
                from base distribution for return periods multiplied by 10**4
                (default=True)

        Returns
        -------
        Creates <retvalsum> dataframe with estimated extreme values for return periods
        and upper and lower confidence bounds, if specified.
        """

        # Parse arguments
        k = kwargs.pop('k', 10 ** 2)
        truncate = kwargs.pop('truncate', True)
        loc = kwargs.pop('loc', False)
        confidence_method = kwargs.pop('confidence_method', 'bootstrap')
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        # Make sure extremes have been extracted
        if not hasattr(self, 'extremes'):
            raise AttributeError(
                'No extremes found. Execute the .get_extremes method before fitting the distribution.'
            )

        # Fit the distribution to the extracted extreme values
        self.distribution = getattr(scipy.stats, distribution)
        if loc:
            self.fit_parameters = self.distribution.fit(self.extremes[self.column] - self.threshold, floc=loc)
        else:
            self.fit_parameters = self.distribution.fit(self.extremes[self.column] - self.threshold)
        def return_value_function(t):
            return self.threshold + self.distribution.isf(
                1/t/self.rate, *self.fit_parameters[:-2],
                loc=self.fit_parameters[-2], scale=self.fit_parameters[-1],
            )

        # Calculate return values based on the fit distribution
        return_periods = np.sort(
            np.unique(
                np.append(
                    np.logspace(-3, 3, num=30),
                    [1, 2, 5, 10, 15, 25, 50, 100, 200, 250, 500, 1000]
                )
            )
        )
        return_values = [return_value_function(t=_t) for _t in return_periods]
        self.retvalsum = pd.DataFrame(data=return_values, index=return_periods, columns=['Return Value'])
        self.retvalsum.index.name = 'Return Period'

        # Estimate confidence intravals
        if confidence_interval:
            if confidence_method == 'montecarlo':
                # Define montecarlo sampling function (returns a sample of return values for return values)
                def montecarlo():
                    # Sample number of extreme values from the Poisson distribution
                    _montecarlo_sample_length = scipy.stats.poisson.rvs(len(self.extremes))
                    _montecarlo_rate = _montecarlo_sample_length / self.N
                    # Sample extreme values
                    _montecarlo_sample = self.distribution.rvs(
                        *self.fit_parameters[:-2], loc=self.fit_parameters[-2], scale=self.fit_parameters[-1],
                        size=_montecarlo_sample_length
                    )
                    # Fit distribution to sampled values
                    if loc:
                        _montecarlo_parameters = self.distribution.fit(_montecarlo_sample, loc=loc)
                    else:
                        _montecarlo_parameters = self.distribution.fit(_montecarlo_sample)
                    return [
                        self.threshold + self.distribution.isf(
                            1 / _t / _montecarlo_rate, *_montecarlo_parameters[:-2],
                            loc=_montecarlo_parameters[-2], scale=_montecarlo_parameters[-1]
                        ) for _t in return_periods
                    ]

                # Perform montecarlo simulation
                simulation_count = 0
                simulated_return_values = []
                if truncate:
                    _upper_limits = np.array([return_value_function(t=10**4 * _t) for _t in return_periods])
                    while simulation_count < k:
                        _simulation = montecarlo()
                        if sum(_simulation > _upper_limits) == 0:
                            simulated_return_values.append(_simulation)
                            simulation_count += 1
                else:
                    while simulation_count < k:
                        _simulation = montecarlo()
                        simulated_return_values.append(_simulation)
                        simulation_count += 1

                # Estimate confidence bounds assuming the error is normally distributed
                filtered = [_x[~np.isnan(_x)] for _x in np.array(simulated_return_values).T]
                moments = [scipy.stats.norm.fit(_x) for _x in filtered]
                intervals = [
                    scipy.stats.norm.interval(alpha=confidence_interval, loc=_x[-2], scale=_x[-1]) for _x in moments
                ]
                self.retvalsum['Lower'] = [_x[0] for _x in intervals]
                self.retvalsum['Upper'] = [_x[1] for _x in intervals]
                self.retvalsum['Sigma'] = [_x[1] for _x in moments]

            elif confidence_method == 'jackknife':
                # TODO - jackknife method (aka leave-one-out)
                raise NotImplementedError

            elif confidence_method == 'bootstrap':
                # Define bootstrap sampling function (returns a sample of return values for return values)
                def bootstrap():
                    # Sample number of extreme values from the Poisson distribution
                    _bootstrap_sample_length = scipy.stats.poisson.rvs(len(self.extremes))
                    _bootstrap_rate = _bootstrap_sample_length / self.N
                    # Resample extreme values
                    _bootstrap_sample = np.random.choice(
                        a=self.extremes[self.column].values,
                        size=_bootstrap_sample_length, replace=True
                    )
                    # Fit distribution to resampled values
                    if loc:
                        _bootstrap_parameters = self.distribution.fit(_bootstrap_sample-self.threshold, loc=loc)
                    else:
                        _bootstrap_parameters = self.distribution.fit(_bootstrap_sample-self.threshold)
                    return [
                        self.threshold + self.distribution.isf(
                            1 / _t / _bootstrap_rate, *_bootstrap_parameters[:-2],
                            loc=_bootstrap_parameters[-2], scale=_bootstrap_parameters[-1]
                        ) for _t in return_periods
                    ]

                # Perform bootstrap simulation
                simulation_count = 0
                simulated_return_values = []
                if truncate:
                    _upper_limits = np.array([return_value_function(t=10 ** 4 * _t) for _t in return_periods])
                    while simulation_count < k:
                        _simulation = bootstrap()
                        if sum(_simulation > _upper_limits) == 0:
                            simulated_return_values.append(_simulation)
                            simulation_count += 1
                else:
                    while simulation_count < k:
                        _simulation = bootstrap()
                        simulated_return_values.append(_simulation)
                        simulation_count += 1

                # Estimate confidence bounds assuming the error is normally distributed
                filtered = [_x[~np.isnan(_x)] for _x in np.array(simulated_return_values).T]
                moments = [scipy.stats.norm.fit(_x) for _x in filtered]
                intervals = [
                    scipy.stats.norm.interval(alpha=confidence_interval, loc=_x[-2], scale=_x[-1]) for _x in moments
                ]
                self.retvalsum['Lower'] = [_x[0] for _x in intervals]
                self.retvalsum['Upper'] = [_x[1] for _x in intervals]
                self.retvalsum['Sigma'] = [_x[1] for _x in moments]

            else:
                raise ValueError('Confidence method {0} not recognized'.format(confidence_method))

    def pdf(self, x):
        """
        Estimates probability density at value <v> using the fitted distribution

        Parameters
        ----------
        x : float or iterable
            Value at which the probability density is estimated

        Returns
        -------
        Depending on x, either estimate or array of estimates of probability densities at <x>
        """
        if hasattr(x, '__iter__'):
            _pdf = []
            for _x in x:
                if _x <= self.threshold:
                    raise ValueError('Extreme value disttribution is not valid for values below the '
                                     'threshold {0:2f} <= {1:.2f}'.format(_x, self.threshold))
                _pdf.append(
                    self.distribution.pdf(
                        _x-self.threshold, *self.fit_parameters[:-2],
                        loc=self.fit_parameters[-2], scale=self.fit_parameters[-1]
                    )
                )
            _pdf = np.array(_pdf)
        else:
            if x <= self.threshold:
                raise ValueError('Extreme value disttribution is not valid for values below the '
                                 'threshold {0:.2f}'.format(self.threshold))
            _pdf = self.distribution.pdf(
                x-self.threshold, *self.fit_parameters[:-2],
                loc=self.fit_parameters[-2], scale=self.fit_parameters[-1]
            )
        return _pdf

    def cdf(self, x):
        """
        Estimates cumulative probability density at value <v> using the fitted distribution

        Parameters
        ----------
        x : float or iterable
            Value at which the cumulative probability density is estimated

        Returns
        -------
        Depending on x, either estimate or array of estimates of cumulative probability densities at <x>
        """
        if hasattr(x, '__iter__'):
            _cdf = []
            for _x in x:
                if _x <= self.threshold:
                    raise ValueError('Extreme value disttribution is not valid for values below the '
                                     'threshold {0:2f} <= {1:.2f}'.format(_x, self.threshold))
                _cdf.append(
                    self.distribution.cdf(
                        _x-self.threshold, *self.fit_parameters[:-2],
                        loc=self.fit_parameters[-2], scale=self.fit_parameters[-1]
                    )
                )
            _cdf = np.array(_cdf)
        else:
            if x <= self.threshold:
                raise ValueError('Extreme value disttribution is not valid for values below the '
                                 'threshold {0:.2f}'.format(self.threshold))
            _cdf = self.distribution.cdf(
                x-self.threshold, *self.fit_parameters[:-2],
                loc=self.fit_parameters[-2], scale=self.fit_parameters[-1]
            )
        return _cdf

    def plot(self, bins=10):
        """
        Plots a summary of the EVA data - pdf, cdf, and distribution fitted to the extracted extreme values.

        Parameters
        ----------
        bins : int
            Number of bins in the PDF and CDF histograms (default=10)

        Returns
        -------
        Generates a matplotlib plot
        """

        # Make sure distribution has been fit
        if not hasattr(self, 'retvalsum'):
            raise AttributeError(
                'No distribution data found. '
                'Execute the .get_extremes and .fit methods before plotting the EVA summary.'
            )
        with plt.style.context('bmh'):
            fig = plt.figure(figsize=(18, 18))
            gs = matplotlib.gridspec.GridSpec(2, 2)
            # Return values plot
            ax1 = fig.add_subplot(gs[0, :])
            ax1.scatter(
                self.extremes['T'], self.extremes[self.column],
                facecolors='None', edgecolors='royalblue', s=20, lw=1,
                label=r'Observed extreme values'
            )
            ax1.plot(
                self.retvalsum.index, self.retvalsum['Return Value'],
                color='orangered', lw=3, ls='-',
                label=r'Fit {0} distribution'.format(self.distribution.name)
            )
            if len(self.retvalsum.columns) > 1:
                ax1.fill_between(
                    self.retvalsum.index.values, self.retvalsum['Upper'].values,
                    self.retvalsum['Lower'].values, alpha=0.3, color='royalblue',
                    label=r'Confidence interval'
                )
            ax1.legend()
            ax1.semilogx()
            ax1.set_ylim(
                0, np.ceil(
                    max(
                        [
                            self.extremes[self.column].values.max() * 1.5,
                            self.retvalsum[self.retvalsum.index == 100.00]['Return Value'].values[0] * 1.1
                        ]
                    )
                )
            )
            # PDF plot
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.hist(
                self.extremes[self.column], bins=bins, density=True,
                color='royalblue', rwidth=.9, alpha=0.5
            )
            ax2.plot(
                np.sort(self.extremes[self.column].values),
                self.pdf(np.sort(self.extremes[self.column].values)),
                color='orangered', lw=3, ls='-'
            )
            ax2.set_ylim(
                0, np.ceil(
                    max(np.histogram(self.extremes[self.column].values, bins=bins, normed=True, density=True)[0]) * 10
                ) / 10
            )
            # CDF plot
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.hist(
                self.extremes[self.column], bins=bins, density=True,
                color='royalblue', cumulative=True, rwidth=.9, alpha=0.5
            )
            ax3.plot(
                np.sort(self.extremes[self.column].values),
                self.cdf(np.sort(self.extremes[self.column].values)),
                color='orangered', lw=3, ls='-'
            )

    def threshold_selection(self, thresholds, method='empirical', **kwargs):
        """

        Parameters
        ----------
        thresholds
        method
        kwargs

        Returns
        -------

        """

        # Parse arguments
        decluster = kwargs.pop('decluster', True)
        r = kwargs.pop('r', 24)
        k = kwargs.pop('k', 10**2)
        confidence_interval = kwargs.pop('confidence_interval', 0.95)
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        # Generate threshold selection assistance data
        if method == 'empirical':
            # 90% rulse
            tres = [
                np.percentile(self.data[self.column].values, 90)
            ]

            # square root method
            k = int(np.sqrt(len(self.data)))
            for threshold in thresholds:
                self.get_extremes(method='POT', threshold=threshold, r=r, decluster=decluster)
                if len(self.extremes) <= k:
                    tres.append(threshold)
                    break
            del self.extremes

            # log method
            k = int(
                len(self.data) ** (2 / 3) / np.log(np.log(len(self.data)))
            )
            for threshold in thresholds:
                self.get_extremes(method='POT', threshold=threshold, r=r, decluster=decluster)
                if len(self.extremes) <= k:
                    tres.append(threshold)
                    break
            del self.extremes

            return pd.DataFrame(data=tres, index=['90% Quantile', 'Squre Root Rule', 'Logarithm Rule'],
                                columns=['Threshold'])

        elif method == 'mean residual life plot':
            mean_residuals = []
            mean_residuals_lower, mean_residuals_upper = [], []
            for threshold in thresholds:
                # Estimate mean residuals
                self.get_extremes(method='POT', threshold=threshold, r=r, decluster=decluster)
                mean_residuals.append(
                    (self.extremes[self.column].values - threshold).mean()
                )

                # Estimate residual confidence bounds using the bootstrap method
                _simulation_count = 0
                _mean_residuals = []
                while _simulation_count < k:
                    _sample = np.random.choice(
                        a=self.extremes[self.column].values,
                        size=len(self.extremes), replace=True
                    )
                    try:
                        _mean_residuals.append(
                            (_sample - threshold).mean()
                        )
                    except TypeError:
                        _mean_residuals.append(_sample)
                    _simulation_count += 1
                _moments = scipy.stats.norm.fit(_mean_residuals)
                _intervals = scipy.stats.norm.interval(alpha=confidence_interval, loc=_moments[-2], scale=_moments[-1])
                mean_residuals_lower.append(_intervals[0])
                mean_residuals_upper.append(_intervals[1])

            # Generate a Pandas DataFrame with mean residual life data
            self.mean_residuals = pd.DataFrame(data=mean_residuals, index=thresholds, columns=['Mean Residual'])
            self.mean_residuals.index.name = 'Threshold'
            self.mean_residuals['Lower'] = mean_residuals_lower
            self.mean_residuals['Upper'] = mean_residuals_upper

            # Prepare the mean residual plot
            with plt.style.context('bmh'):
                plt.figure(figsize=(18, 8))
                plt.plot(
                    thresholds, mean_residuals,
                    color='orangered', lw=3, ls='-',
                    label=r'Mean residuals'
                )
                plt.fill_between(
                    thresholds, mean_residuals_upper,
                    mean_residuals_lower, alpha=0.3, color='royalblue',
                    label=r'{0:.0f}% confidence interval'.format(confidence_interval*100)
                )
                plt.legend()
                plt.ylim(
                    0,
                    np.ceil(
                       max(mean_residuals) * 1.5
                    )
                )


# Test
# import os
# input_folder = r'D:\Work folders\desktop projects\3 NPS STLI\1 Data\1 Wind\2 LCD Station 14732, La Guardia Airport'
# data = pd.read_pickle(os.path.join(input_folder, 'Wind, Station 14732, 1957-2018.pyc'))
# data = data.dropna()
# eve = EVA(data)
# eve.get_extremes(threshold=35)
# eve.fit(
#     distribution='genpareto', confidence_interval=0.95, confidence_method='bootstrap',
#     k=10**2, loc=0, truncate=True
# )
# eve.plot(bins=10)
#
# eve.threshold_selection(method='empirical', thresholds=np.arange(30, 60))
# eve.threshold_selection(method='mean residual life plot', thresholds=np.arange(30, 60, 1))


# def par_stab_plot(self, u, distribution='GPD', decluster=True, dmethod='naive',
#                   r=24, save_path=None, name='_DATA_SOURCE_'):
#     """
#     Generates a parameter stability plot for the a range of thresholds u.
#     :param u: list or array
#         List of threshold values.
#     :param decluster: bool
#         Use run method to decluster data 9default = True)
#     :param r: float
#         Run lengths (hours), specify if decluster=True.
#     :param save_path: str
#         Path to save folder.
#     :param name: str
#         File save name.
#     :param dmethod: str
#         Decluster method (Default = 'naive')
#     :param distribution: str
#         Distribution name
#     """
#     u = np.array(u)
#     if u.max() > self.data[self.col].max():
#         u = u[u <= self.data[self.col].max()]
#     if distribution == 'GPD':
#         fits = []
#         for tres in u:
#             self.get_extremes(method='POT', threshold=tres, r=r, decluster=decluster, dmethod=dmethod)
#             extremes_local = self.extremes[self.col].values - tres
#             fits.append(scipy.stats.genpareto.fit(extremes_local))
#         shapes = [x[0] for x in fits]
#         scales = [x[2] for x in fits]
#         # scales_mod = [scales[i] - shapes[i] * u[i] for i in range(len(u))]
#         scales_mod = scales
#         with plt.style.context('bmh'):
#             plt.figure(figsize=(16, 8))
#             plt.subplot(1, 2, 1)
#             plt.plot(u, shapes, lw=2, color='orangered', label=r'Shape Parameter')
#             plt.xlabel(r'Threshold Value')
#             plt.ylabel(r'Shape Parameter')
#             plt.subplot(1, 2, 2)
#             plt.plot(u, scales_mod, lw=2, color='orangered', label=r'Scale Parameter')
#             plt.xlabel(r'Threshold Value')
#             plt.ylabel(r'Scale Parameter')
#             plt.suptitle(r'{} Parameter Stability Plot'.format(name))
#         if not save_path:
#             plt.show()
#         else:
#             plt.savefig(os.path.join(save_path, '{} Parameter Stability Plot.png'.format(name)),
#                         bbox_inches='tight', dpi=600)
#             plt.close()
#     else:
#         print('The {} distribution is not yet implemented for this method'.format(distribution))

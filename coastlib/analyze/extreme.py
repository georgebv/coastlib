import datetime
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec
import numpy as np
import pandas as pd
import scipy.stats


input_folder = r'D:\Work folders\desktop projects\3 NPS STLI\1 Data\1 Wind\2 LCD Station 14732, La Guardia Airport'
data = pd.read_pickle(os.path.join(input_folder, 'Wind, Station 14732, 1957-2018.pyc'))
data = data.dropna()
eve = EVA(data)
eve.get_extremes(threshold=30)
eve.fit(
    distribution='exponweib', confidence_interval=0.95, confidence_method='bootstrap', k=10**2, truncate=True, floc=0
)
eve.plot(bins=10)


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

    4   Use the <.plot> method to get a quick summary

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

        Parameters
        ----------
        method : str
            Peak extraction method. POT for peaks over threshold and AM for annual maxima.

        kwargs
            decluster : bool
                If method is POT only: decluster checks if extremes are declustered
            threshold : float
                If method is POT only: threshold for extreme value extraction
            r : float (hours, default=24)
                If method is POT only: minimum distance between events for them to be considered independent

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

    def fit(self, distribution='genpareto', confidence_interval=None, floc=False, **kwargs):
        """
        Fits <distribution> to extracted extreme values. Creates <retvalsum> dataframe with
        estimated extreme values for return periods and upper and lower confidence bounds, if specified.

        Parameters
        ----------
        distribution : str
            Scipy disttribution name (default='genpareto')
        confidence_interval : float
            Confidence interval width, should be confidence_interval<1 or None. Not estimated if None (default=None)
        floc : float
            Fixed location parameter value. Location parameter is estimated if None is passed (default=None)
        kwargs
            confidence_method : str
                Confidence interval estimation methods (default='bootstrap'):
                    -'montecarlo' - montecarlo method. Generates poisson-distributed sized samples
                        from the fitted distribution
                    -'jackknife' - jackknife method. Generates leave-one-out samples
                        from the original extracted extreme values
                    -'bootstrap' - bootstrap method. Generates poisson-distributed sized samples
                        from the original extracted extreme values with replacement
            k : int
                Number of distributions, !!!highly affect performance!!! (default=10**2)

        Returns
        -------
        Creates <retvalsum> dataframe with estimated extreme values for return periods
        and upper and lower confidence bounds, if specified.
        """
        # Make sure extremes have been extracted
        if not hasattr(self, 'extremes'):
            raise AttributeError(
                'No extremes found. Execute the .get_extremes method before fitting the distribution.'
            )

        # Fit the distribution to the extracted extreme values
        self.distribution = getattr(scipy.stats, distribution)
        if floc:
            self.fit_parameters = self.distribution.fit(self.extremes[self.column] - self.threshold, floc=floc)
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
            confidence_method = kwargs.pop('confidence_method', 'bootstrap')
            if confidence_method == 'montecarlo':
                # Parse montecarlo method arguments
                k = kwargs.pop('k', 10**2)
                truncate = kwargs.pop('truncate', True)
                assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

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
                    if floc:
                        _montecarlo_parameters = self.distribution.fit(_montecarlo_sample, floc=floc)
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
                # Parse montecarlo method arguments
                k = kwargs.pop('k', 10 ** 2)
                truncate = kwargs.pop('truncate', False)
                assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

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
                    if floc:
                        _bootstrap_parameters = self.distribution.fit(_bootstrap_sample-self.threshold, floc=floc)
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
                self.extremes['T'], self.extremes[eve.column],
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
                    label=r'95% confidence interval'
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


class EVAOLD:
    """
    Extreme Value Analysis class. Takes a Pandas DataFrame with values. Extracts extreme values.
    Assists with threshold value selection. Fits data to distributions (GPD).
    Returns extreme values' return periods. Generates data plots.

    Workflow:
        for dataframe <df> with values under column 'Hs'
        ~$ eve = EVA(df, col='Hs')
        use pot_residuals, empirical_threshold, par_stab_plot to assist with threshold selection
        ~$ eve.get_extremes(<parameters>)  this will parse extreme values from given data
        ~$ eve.fit(<parameters>)  this will fit a distrivution to the parsed extremes
        ~$ eve.ret_val_plot(<parameters>)  this will produce a plot of extremes with a fit
        eve.retvalsum and eve.extremes will have all the data necessary for a report
    """

    def __init__(self, df, col=None, discontinuous=False):
        """
        Mandatory inputs
        ================
        df : DataFrame or Series
            Pandas DataFrame or Series object with column <col> containing values and indexes as datetime

        Optional inputs
        ===============
        col : str (default=None, takes first column)
            Column name for the variable of interest in <df> (i.e. 'Hs' or 'WS')
        """

        if not isinstance(df, pd.DataFrame):
            try:
                self.data = df.to_frame()
            except AttributeError:
                raise TypeError('Invalid data type in <df>.'
                                ' EVA takes only Pandas DataFrame or Series objects.')
        else:
            self.data = df
        self.data.sort_index(inplace=True)

        if col:
            self.col = col
        else:
            self.col = df.columns[0]

        # Calculate acuatl number of years in data
        years = np.unique(self.data.index.year)
        # Get a list of years between min and max years from the series
        years_all = np.arange(years.min(), years.max()+1, 1)

        self.N = len(years)
        if discontinuous:
            self.N = len(years_all)

        if self.N != len(years_all):
            missing = [year for year in years_all if year not in years]
            print(
                '\n\nData is not continuous!\nMissing years {}\n'
                'Set <dicontinuous=True> to account for all years, '
                'assuming there were NO peaks in the missing years.\n'
                'Without this option turned on, extreme events might be\n'
                'significantly overestimated (conservative results)'.format(missing)
            )

    def __repr__(self):

        try:
            lev = len(self.extremes)
        except AttributeError:
            lev = 'not extracted'

        try:
            dis = self.distribution
        except AttributeError:
            dis = 'not assigned'

        try:
            em = self.method
        except AttributeError:
            em = 'not assigned'

        return 'EVA(col={col})\n' \
               '\n' \
               '         Number of years -> {yr}\n' \
               'Number of extreme events -> {lev}\n' \
               '       Extraction method -> {em}\n' \
               '       Distribution used -> {dis}'.format(col=self.col, yr=self.N, lev=lev, dis=dis, em=em)

    def get_extremes(self, method='POT', **kwargs):
        """
        Extracts extreme values and places them in <self.extremes>
        Uses Weibull plotting position

        Optional inputs
        ===============
        method : str (default='POT')
            Extraction method. 'POT' for Peaks Over Threshold, 'BM' for Block Maxima
        decluster : bool (default=False)
            Specify if extreme values are declustered (POT method only)

        dmethod : str (default='naive')
            Declustering method. 'naive' method linearly declusters the series (POT method only)
        threshold : float (default=10)
            Threshold value (POT method only)
        r : float (default=24)
            Minimum independent event distance (hours) (POT method only)

        block : timedelta (default=1)
            Block size (years) (BM method only)
        """

        self.method = method

        if method == 'POT':
            decluster = kwargs.pop('decluster', True)
            dmethod = kwargs.pop('dmethod', 'naive')
            try:
                self.threshold = kwargs.pop('threshold')
            except KeyError:
                raise ValueError('<threshold> is required for the POT method')
            r = datetime.timedelta(hours=kwargs.pop('r', 24))
            assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

            self.extremes = self.data[self.data[self.col] > self.threshold]
            if decluster:
                if dmethod == 'naive':
                    # Set first peak to 0th element of the raw peaks-over-threshold array
                    # This 0th element is assumed to be a local maxima
                    indexes, values = [self.extremes.index[0]], [self.extremes[self.col][0]]
                    for i, index, value in zip(
                            range(len(self.extremes.index)),
                            self.extremes.index,
                            self.extremes[self.col]
                    ):
                        # Check for time condition
                        if index - indexes[-1] >= r:
                            indexes.append(index)
                            values.append(value)
                        # If its in the same cluster (failed time condition),
                        # check if its larger than current peak for this cluster
                        # = ensures the latest peak being captured (reduces total number of clusters)
                        elif value >= values[-1]:
                            indexes[-1] = index
                            values[-1] = value
                else:
                    raise ValueError('Method {} is not yet implemented'.format(dmethod))
                self.extremes = pd.DataFrame(data=values, index=indexes, columns=[self.col])

        elif method == 'BM':
            indexes, values = [self.data.index[0]], [self.data[self.col][0]]
            for index, value in zip(self.data.index, self.data[self.col]):
                if index.year == indexes[-1].year:
                    if value > values[-1]:
                        indexes[-1] = index
                        values[-1] = value
                else:
                    indexes.extend([index])
                    values.extend([value])
            self.extremes = pd.DataFrame(data=values, index=indexes, columns=[self.col])

        else:
            raise ValueError('Unrecognized extremes parsing method {}. Use POT or BM methods.'.format(method))

        # Weibull plotting position
        self.extremes.sort_values(by=self.col, ascending=True, inplace=True)
        self.rate = len(self.extremes) / self.N  # extreme events per year
        cdf = np.arange(1, len(self.extremes) + 1) / (len(self.extremes) + 1)  # m/N+1 (rank/total+1)
        icdf = 1 - cdf  # inverse cdf (probability of exceedance)
        self.extremes['T'] = 1 / (icdf * self.rate)  # annual probability to return period
        self.extremes.sort_index(inplace=True)

    def pot_residuals(self, u, decluster=True, r=24, save_path=None, dmethod='naive', name='_DATA_SOURCE_'):
        """
        Calculates mean residual life values for different threshold values.

        :param u: list
            List of threshold to be tested
        :param save_path: str
            Path to folder. Default = None.
        :param name: str
            Plot name.
        :param decluster: bool
            Decluster data using the run method.
        :param r: float
            Decluster run length (Default = 24 hours).
        :param dmethod: str
            Decluster method (Default = 'naive')
        :return:
        """

        u = np.array(u)
        if u.max() > self.data[self.col].max():
            u = u[u <= self.data[self.col].max()]
        if decluster:
            nu, res_ex_sum = [], []
            for i in range(len(u)):
                self.get_extremes(method='POT', threshold=u[i], r=r, decluster=True, dmethod=dmethod)
                nu.extend([len(self.extremes)])
                res_ex_sum.extend([self.extremes[self.col].values - u[i]])
        else:
            nu = [len(self.data[self.data[self.col] >= i]) for i in u]
            res_ex_sum = [(self.data[self.data[self.col] >= _u][self.col] - _u).values for _u in u]
        residuals = [(sum(res_ex_sum[i]) / nu[i]) for i in range(len(u))]
        intervals = [
            scipy.stats.norm.interval(
                0.95, loc=res_ex_sum[i].mean(), scale=res_ex_sum[i].std() / len(res_ex_sum[i])
            )
            for i in range(len(u))
        ]
        intervals_u = [interval[0] for interval in intervals]
        intervals_l = [interval[1] for interval in intervals]
        with plt.style.context('bmh'):
            plt.subplot(1, 1, 1)
            plt.plot(u, residuals, lw=2, color='orangered', label=r'Mean Residual Life')
            plt.fill_between(u, intervals_u, intervals_l, alpha=0.3, color='royalblue',
                             label=r'95% confidence interval')
            plt.xlabel(r'Threshold Value')
            plt.ylabel(r'Mean residual Life')
            plt.title(r'{} Mean Residual Life Plot'.format(name))
            plt.legend()
        if not save_path:
            plt.show()
        else:
            plt.savefig(os.path.join(save_path, '{} Mean Residual Life.png'.format(name)),
                        bbox_inches='tight', dpi=300)
            plt.close()

    def empirical_threshold(self, decluster=False, dmethod='naive', r=24, u_step=0.1, u_start=0):
        """
        Get exmpirical threshold extimates for 3 methods: 90% percentile value,
        square root method, log-method.

        :param decluster:
            Determine if declustering is used for estimating thresholds.
            Very computationally intensive.
        :param r:
            Declustering run length parameter.
        :param u_step:
            Threshold precision.
        :param u_start:
            Starting threshold for search (should be below the lowest expected value).
        :param dmethod: str
            Decluster method (Default = 'naive')
        :return:
            DataFrame with threshold summary.
        """

        # 90% rulse
        tres = [np.percentile(self.data[self.col].values, 90)]

        # square root method
        k = int(np.sqrt(len(self.data)))
        u = u_start
        self.get_extremes(method='POT', threshold=u, r=r, decluster=decluster, dmethod=dmethod)
        while len(self.extremes) > k:
            u += u_step
            self.get_extremes(method='POT', threshold=u, r=r, decluster=decluster, dmethod=dmethod)
        tres.extend([u])

        # log method
        k = int((len(self.data) ** (2 / 3)) / np.log(np.log(len(self.data))))
        u = u_start
        self.get_extremes(method='POT', threshold=u, r=r, decluster=decluster, dmethod=dmethod)
        while len(self.extremes) > k:
            u += u_step
            self.get_extremes(method='POT', threshold=u, r=r, decluster=decluster, dmethod=dmethod)
        tres.extend([u])

        return pd.DataFrame(data=tres, index=['90% Quantile', 'Squre Root Rule', 'Logarithm Rule'],
                            columns=['Threshold'])

    def par_stab_plot(self, u, distribution='GPD', decluster=True, dmethod='naive',
                      r=24, save_path=None, name='_DATA_SOURCE_'):
        """
        Generates a parameter stability plot for the a range of thresholds u.
        :param u: list or array
            List of threshold values.
        :param decluster: bool
            Use run method to decluster data 9default = True)
        :param r: float
            Run lengths (hours), specify if decluster=True.
        :param save_path: str
            Path to save folder.
        :param name: str
            File save name.
        :param dmethod: str
            Decluster method (Default = 'naive')
        :param distribution: str
            Distribution name
        """
        # TODO - buggy method (scales and shapes are weird)
        u = np.array(u)
        if u.max() > self.data[self.col].max():
            u = u[u <= self.data[self.col].max()]
        if distribution == 'GPD':
            fits = []
            for tres in u:
                self.get_extremes(method='POT', threshold=tres, r=r, decluster=decluster, dmethod=dmethod)
                extremes_local = self.extremes[self.col].values - tres
                fits.append(scipy.stats.genpareto.fit(extremes_local))
            shapes = [x[0] for x in fits]
            scales = [x[2] for x in fits]
            # scales_mod = [scales[i] - shapes[i] * u[i] for i in range(len(u))]
            scales_mod = scales
            with plt.style.context('bmh'):
                plt.figure(figsize=(16, 8))
                plt.subplot(1, 2, 1)
                plt.plot(u, shapes, lw=2, color='orangered', label=r'Shape Parameter')
                plt.xlabel(r'Threshold Value')
                plt.ylabel(r'Shape Parameter')
                plt.subplot(1, 2, 2)
                plt.plot(u, scales_mod, lw=2, color='orangered', label=r'Scale Parameter')
                plt.xlabel(r'Threshold Value')
                plt.ylabel(r'Scale Parameter')
                plt.suptitle(r'{} Parameter Stability Plot'.format(name))
            if not save_path:
                plt.show()
            else:
                plt.savefig(os.path.join(save_path, '{} Parameter Stability Plot.png'.format(name)),
                            bbox_inches='tight', dpi=600)
                plt.close()
        else:
            print('The {} distribution is not yet implemented for this method'.format(distribution))

    def fit(self, distribution='GPD', confidence=0.95, k=10**2, trunc=True, method='montecarlo'):
        """
        Implemented: GEV, GPD, Weibull, Log-normal, Pearson 3
        Fits distribution to data and generates a summary dataframe (required for plots)

        Optional inputs
        ===============
        distribution : str (default='GPD')
            Distribution name . Available: GPD, GEV, Gumbel, Wibull, Log-normal, Pearson 3
        confidence : bool or float [0:1] (default=0.95)
            if float, used as confidence interval; if False, avoids this altogether
            Calculate 95% confidence limits using Monte Carlo simulation
            !!!!    (WARNING! Might be time consuming for large k)    !!!!
            Be cautious with interpreting the 95% confidence limits.
        k : int (default=10**2)
            Number of montecarlo simulations. Has significant performance impact
        trunc : bool (default=True)
            If True, ignores unreasonable (extreme) values generated by montecarlo method
            Runs until <k> is saturated
        :return: DataFrame
            self.retvalsum summary dataframe with fitted distribution and 95% confidence limits.
        """

        self.distribution = distribution

        if self.method == 'BM':
            self.threshold = None

        # Define the <ret_val> function for selected distibution. This function takes
        # fit parameters (scale, loc,..) and returns <return values> for return periods <t>
        if self.distribution == 'GPD':
            if self.method != 'POT':
                raise ValueError('GPD distribution is applicable only with the POT method')
            self.fit_parameters = scipy.stats.genpareto.fit(self.extremes[self.col].values - self.threshold)
            def ret_val(t):
                return self.threshold + scipy.stats.genpareto.ppf(
                    1 - 1 / (self.rate * t),
                    c=self.fit_parameters[0], loc=self.fit_parameters[1], scale=self.fit_parameters[2]
                )

        elif self.distribution == 'GEV':
            if self.method != 'BM':
                raise ValueError('GEV distribution is applicable only with the BM method')
            self.fit_parameters = scipy.stats.genextreme.fit(self.extremes[self.col].values)
            def ret_val(t):
                return scipy.stats.genextreme.ppf(
                    1 - 1 / (self.rate * t),
                    c=self.fit_parameters[0], loc=self.fit_parameters[1], scale=self.fit_parameters[2]
                )

        elif self.distribution == 'Weibull':
            if self.method != 'POT':
                raise ValueError('Weibull distribution is applicable only with the POT method')
            self.fit_parameters = scipy.stats.invweibull.fit(self.extremes[self.col].values - self.threshold)
            def ret_val(t):
                return self.threshold + scipy.stats.invweibull.ppf(
                    1 - 1 / (self.rate * t),
                    c=self.fit_parameters[0], loc=self.fit_parameters[1], scale=self.fit_parameters[2]
                )

        elif self.distribution == 'Log-normal':
            if self.method == 'POT':
                self.fit_parameters = scipy.stats.lognorm.fit(self.extremes[self.col].values - self.threshold)
                def ret_val(t):
                    return self.threshold + scipy.stats.lognorm.ppf(
                        1 - 1 / (self.rate * t),
                        s=self.fit_parameters[0], loc=self.fit_parameters[1], scale=self.fit_parameters[2]
                    )
            else:
                self.fit_parameters = scipy.stats.lognorm.fit(self.extremes[self.col].values)
                def ret_val(t):
                    return scipy.stats.lognorm.ppf(
                        1 - 1 / (self.rate * t),
                        s=self.fit_parameters[0], loc=self.fit_parameters[1], scale=self.fit_parameters[2]
                    )

        elif self.distribution == 'Pearson 3':
            if self.method == 'POT':
                self.fit_parameters = scipy.stats.pearson3.fit(self.extremes[self.col].values - self.threshold)
                def ret_val(t):
                    return self.threshold + scipy.stats.pearson3.ppf(
                        1 - 1 / (self.rate * t),
                        skew=self.fit_parameters[0], loc=self.fit_parameters[1], scale=self.fit_parameters[2]
                    )
            else:
                self.fit_parameters = scipy.stats.pearson3.fit(self.extremes[self.col].values)
                def ret_val(t):
                    return scipy.stats.pearson3.ppf(
                        1 - 1 / (self.rate * t),
                        skew=self.fit_parameters[0], loc=self.fit_parameters[1], scale=self.fit_parameters[2]
                    )

        # TODO =================================================================================
        # TODO - seems good, but test (expon, fretchet)
        elif self.distribution == 'Gumbel':
            if self.method != 'BM':
                raise ValueError('Gumbel distribution is applicable only with the BM method')
            self.fit_parameters = scipy.stats.gumbel_r.fit(self.extremes[self.col].values)
            def ret_val(t):
                return scipy.stats.gumbel_r.ppf(
                    1 - 1 / (self.rate * t),
                    loc=self.fit_parameters[0], scale=self.fit_parameters[1]
                )
        else:
            raise ValueError('Distribution type {} not recognized'.format(self.distribution))
        # TODO =================================================================================

        # Return periods equally spaced on log scale from 0.1y to 1000y
        rp = np.unique(np.append(np.logspace(0, 3, num=30), [1/12, 2, 5, 10, 25, 50, 100, 200, 500]))
        # rp = np.unique(np.append(rp, self.extremes['T'].values))
        rp = np.sort(rp)
        rv = [ret_val(x) for x in rp]
        self.retvalsum = pd.DataFrame(data=rv, index=rp, columns=['Return Value'])
        self.retvalsum.index.name = 'Return Period'

        if confidence:

            # Collect statistics using defined montecarlo
            if method == 'montecarlo':
                # Define montecarlo return values generator
                if self.distribution == 'GPD':
                    def montefit():
                        _lex = scipy.stats.poisson.rvs(len(self.extremes))
                        _sample = scipy.stats.genpareto.rvs(
                            c=self.fit_parameters[0], loc=self.fit_parameters[1], scale=self.fit_parameters[2],
                            size=_lex
                        )
                        _param = scipy.stats.genpareto.fit(_sample)  # floc=parameters[1]
                        return self.threshold + scipy.stats.genpareto.ppf(
                            1 - 1 / (self.rate * rp),
                            c=_param[0], loc=_param[1], scale=_param[2]
                        )

                elif self.distribution == 'GEV':
                    def montefit():
                        _lex = self.N
                        _sample = scipy.stats.genextreme.rvs(
                            c=self.fit_parameters[0], loc=self.fit_parameters[1], scale=self.fit_parameters[2],
                            size=_lex
                        )
                        _param = scipy.stats.genextreme.fit(_sample)  # floc=parameters[1]
                        return scipy.stats.genextreme.ppf(
                            1 - 1 / (self.rate * rp),
                            c=_param[0], loc=_param[1], scale=_param[2]
                        )

                elif self.distribution == 'Weibull':
                    def montefit():
                        _lex = scipy.stats.poisson.rvs(len(self.extremes))
                        _sample = scipy.stats.invweibull.rvs(
                            c=self.fit_parameters[0], loc=self.fit_parameters[1], scale=self.fit_parameters[2],
                            size=_lex
                        )
                        _param = scipy.stats.invweibull.fit(_sample)
                        return self.threshold + scipy.stats.invweibull.ppf(
                            1 - 1 / (self.rate * rp),
                            c=_param[0], loc=_param[1], scale=_param[2]
                        )

                elif self.distribution == 'Log-normal':
                    def montefit():
                        if self.method == 'POT':
                            _lex = scipy.stats.poisson.rvs(len(self.extremes))
                        else:
                            _lex = self.N
                        _sample = scipy.stats.lognorm.rvs(
                            s=self.fit_parameters[0], loc=self.fit_parameters[1], scale=self.fit_parameters[2],
                            size=_lex
                        )
                        _param = scipy.stats.lognorm.fit(_sample)
                        if self.method == 'POT':
                            return self.threshold + scipy.stats.lognorm.ppf(
                                1 - 1 / (self.rate * rp),
                                s=_param[0], loc=_param[1], scale=_param[2]
                            )
                        else:
                            return scipy.stats.lognorm.ppf(
                                1 - 1 / (self.rate * rp),
                                s=_param[0], loc=_param[1], scale=_param[2]
                            )

                elif self.distribution == 'Pearson 3':
                    def montefit():
                        if self.method == 'POT':
                            _lex = scipy.stats.poisson.rvs(len(self.extremes))
                        else:
                            _lex = self.N
                        _sample = scipy.stats.pearson3.rvs(
                            skew=self.fit_parameters[0], loc=self.fit_parameters[1], scale=self.fit_parameters[2],
                            size=_lex
                        )
                        _param = scipy.stats.pearson3.fit(_sample)
                        if self.method == 'POT':
                            return self.threshold + scipy.stats.pearson3.ppf(
                                1 - 1 / (self.rate * rp),
                                skew=_param[0], loc=_param[1], scale=_param[2]
                            )
                        else:
                            return scipy.stats.pearson3.ppf(
                                1 - 1 / (self.rate * rp),
                                skew=_param[0], loc=_param[1], scale=_param[2]
                            )

                else:
                    raise ValueError('Montecarlo method is not implemented for {} distribution'.
                                     format(self.distribution))

                sims = 0
                mrv = []
                if trunc:
                    uplims = ret_val(10**4 * rp)
                    while sims < k:
                        x = montefit()
                        if sum(x > uplims) == 0:
                            mrv.append(x)
                            sims += 1
                else:
                    while sims < k:
                        mrv.append(montefit())
                        sims += 1
                # Using normal distribution, get <confidence> confidence bounds
                filtered = [x[~np.isnan(x)] for x in np.array(mrv).T]
                moments = [scipy.stats.norm.fit(x) for x in filtered]
                intervals = [scipy.stats.norm.interval(alpha=confidence, loc=x[0], scale=x[1]) for x in moments]
                self.retvalsum['Lower'] = [x[0] for x in intervals]
                self.retvalsum['Upper'] = [x[1] for x in intervals]
                self.retvalsum['Sigma'] = [x[1] for x in moments]

            elif method == 'jacknife':
                # TODO - verify this method works as intended
                # Define jacknife return values generator
                if self.distribution == 'GPD':
                    def jacknife(_i, _lex, _idx):
                        _sample = self.extremes[self.col].values[_idx != _i]
                        _param = scipy.stats.genpareto.fit(_sample - self.threshold)
                        return self.threshold + scipy.stats.genpareto.ppf(
                            1 - 1 / (self.rate * rp),
                            c=_param[0], loc=_param[1], scale=_param[2]
                        )

                elif self.distribution == 'GEV':
                    def jacknife(_i, _lex, _idx):
                        _sample = self.extremes[self.col].values[_idx != _i]
                        _param = scipy.stats.genextreme.fit(_sample)
                        return scipy.stats.genextreme.ppf(
                            1 - 1 / (self.rate * rp),
                            c=_param[0], loc=_param[1], scale=_param[2]
                        )

                elif self.distribution == 'Weibull':
                    def jacknife(_i, _lex, _idx):
                        _sample = self.extremes[self.col].values[_idx != _i]
                        _param = scipy.stats.invweibull.fit(_sample)
                        return self.threshold + scipy.stats.invweibull.ppf(
                            1 - 1 / (self.rate * rp),
                            c=_param[0], loc=_param[1], scale=_param[2]
                        )

                elif self.distribution == 'Log-normal':
                    def jacknife(_i, _lex, _idx):
                        _sample = self.extremes[self.col].values[_idx != _i]
                        _param = scipy.stats.lognorm.fit(_sample)
                        if self.method == 'POT':
                            return self.threshold + scipy.stats.lognorm.ppf(
                                1 - 1 / (self.rate * rp),
                                s=_param[0], loc=_param[1], scale=_param[2]
                            )
                        else:
                            return scipy.stats.lognorm.ppf(
                                1 - 1 / (self.rate * rp),
                                s=_param[0], loc=_param[1], scale=_param[2]
                            )

                elif self.distribution == 'Pearson 3':
                    def jacknife(_i, _lex, _idx):
                        _sample = self.extremes[self.col].values[_idx != _i]
                        _param = scipy.stats.pearson3.fit(_sample)
                        if self.method == 'POT':
                            return self.threshold + scipy.stats.pearson3.ppf(
                                1 - 1 / (self.rate * rp),
                                skew=_param[0], loc=_param[1], scale=_param[2]
                            )
                        else:
                            return scipy.stats.pearson3.ppf(
                                1 - 1 / (self.rate * rp),
                                skew=_param[0], loc=_param[1], scale=_param[2]
                            )

                else:
                    raise ValueError('Jackinfe method is not implemented for {} distribution'.
                                     format(self.distribution))

                lex = len(self.extremes)
                idx = np.arange(lex)
                mrv = []
                for i in range(lex):
                    mrv.append(jacknife(_i=i, _lex=lex, _idx=idx))
                # Using normal distribution, get <confidence> confidence bounds
                filtered = np.array([x[~np.isnan(x)] for x in np.array(mrv).T])
                x_ = np.array([np.nansum(row) for row in filtered]) / lex
                x_t = np.array([filtered[i] * lex - x_[i] * (lex - 1) for i in range(len(filtered))])
                _x_t = np.array([np.nanmean(row) for row in x_t])
                std = np.array([((lex - 1) / lex) * np.nansum((filtered[i] - x_[i]) ** 2)
                                for i in range(len(filtered))]) ** .5
                intervals = [scipy.stats.norm.interval(alpha=confidence, loc=loc, scale=scale)
                             for loc, scale in zip(_x_t, std)]
                # moments = [scipy.stats.norm.fit(x) for x in filtered]
                # intervals = [scipy.stats.norm.interval(alpha=confidence, loc=x[0], scale=x[1]) for x in moments]
                self.retvalsum['Lower'] = [x[0] for x in intervals]
                self.retvalsum['Upper'] = [x[1] for x in intervals]

            else:
                raise ValueError('Method {} is not recognized'.format(method))

        self.retvalsum.dropna(inplace=True)

    def ret_val_plot(self, confidence=False, save_path=None, name='_DATA_SOURCE_', **kwargs):
        """
        Creates return value plot (return periods vs return values)

        :param confidence: bool
            True if confidence limits were calculated in the .fit() method.
        :param save_path: str
        :param name: str
        :param kwargs:
            unit: str
                Return value unit (i.e. m/s) default = unit
            ylim: tuple
                Y axis limits (to avoid showing entire confidence limit range). Default=(0, ReturnValues.max()).
        :return:
        """

        # TODO - method is broken and unreliable
        unit = kwargs.pop('unit', 'unit')
        ylim = kwargs.pop('ylim', [self.extremes[self.col].values.min(), self.extremes[self.col].values.max()])
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))
        with plt.style.context('bmh'):
            plt.figure(figsize=(16, 8))
            plt.subplot(1, 1, 1)
            plt.scatter(self.extremes['T'].values, self.extremes[self.col].values, s=20, linewidths=1,
                        marker='o', facecolor='None', edgecolors='royalblue', label=r'Extreme Values')
            plt.plot(self.retvalsum.index.values, self.retvalsum['Return Value'].values,
                     lw=2, color='orangered', label=r'{} Fit'.format(self.distribution))
            if confidence:
                plt.fill_between(self.retvalsum.index.values, self.retvalsum['Upper'].values,
                                 self.retvalsum['Lower'].values, alpha=0.3, color='royalblue',
                                 label=r'95% confidence interval')
            plt.xscale('log')
            plt.xlabel(r'Return Period [years]')
            plt.ylabel(r'Return Value [{0}]'.format(unit))
            plt.title(r'{0} {1} Return Values Plot'.format(name, self.distribution))
            plt.xlim((self.retvalsum.index.values.min(), self.retvalsum.index.values.max()))
            plt.ylim(ylim)
            plt.legend(loc=2)
            plt.grid(linestyle='--', which='minor')
            plt.grid(linestyle='-', which='major')
            if not save_path:
                plt.show()
            else:
                plt.savefig(save_path + '\{0} {1} Return Values Plot.png'.format(name, self.distribution),
                            bbox_inches='tight', dpi=300)
                plt.close()

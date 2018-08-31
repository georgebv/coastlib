import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


class EVA:

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
                raise AttributeError('Column {0} cannot be accessed. Check spelling'.format(column))
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
                'Without this option turned on, extreme events might be\n'
                'significantly overestimated (conservative results)'.format(missing)
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

        Returns
        -------
        Creates a self.extremes dataframe with extreme values and return periods determined using
        the Weibull plotting position P=m/(N+1)
        """


class EVA_old:
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

    def dens_fit_plot(self, distribution='GPD'):
        """
        Probability density plot. Histogram of extremes with fit overlay.

        :param distribution:
        :return:
        """

        # TODO - implemented
        print('Not yet implemented')

import pickle

import corner
import emcee
import matplotlib.pyplot as plt
import matplotlib.ticker
import mpmath
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm

import coastlib.math.derivatives
import coastlib.stats.distributions


# Helper function used to handle quantiles of empty arrays
def empty_quantile(array, *args, **kwargs):
    if len(array) > 0:
        return np.nanquantile(array, *args, **kwargs)
    else:
        return np.nan


class EVA:
    """
    Initializes the EVA class instance by taking a <dataframe> with values in <column> to analyze.
    Extracts extreme values. Provides assistance in threshold value selection for the POT method.

    Estimates parameters of distributions for given data using Maximum Likelihood Estimate (MLE)
    or estimates posterior distributions of parameters of distributions using Markov chain Monte Carlo (MCMC).

    For given return periods gives estimates of return values and associated confidence intervals.
    Generates various statistical plots such as return value plot and QQ/PP plots.
    Provides multiple goodness-of-fit (GOF) statistics and tests.

    Parameters
    ----------
    dataframe : pd.DataFrame or pd.Series
        Pandas Dataframe or Series object containing data to be analyzed.
        Must have index array of type pd.DatetimeIndex.
    column : str or int, optional
        Name or index of column in <dataframe> with data to be analyzed.
        By default is <None> and takes first (0'th index) column from <dataframe>.
    block_size : float, optional
        Block size in days. Used to determine number of blocks in data (default=365.2425, one Gregorian year).
        Block size is used to estimate probabilities (return periods for observed data) for all methods
        and to extract extreme events in the 'Block Maxima' method.
        By default, it is one Gregorian year and results in return periods having units of years,
        i.e. a 100-<block_size> event by default is a 100-year return period event.
        Weekly would be <block_size=7> and monthly would be <block_size=365.2425/12>.
    gap_length : float, optional
        Gap length in hours. Gaps larger than <gap_length> are excluded when calculating total
        number of blocks of <block_size> in <dataframe>. Set to None to calculate number of blocks
        as "(last_date - first_date) / block_size". Default is 24 hours.
        It is also used in Block Maxima extreme value extraction method to get boundaries of blocks.

    Public Attributes
    -----------------
    self.__init__()
        self.dataframe : pd.DataFrame
        self.column : str
        self.block_size : float
        self.gap_length : float
        self.number_of_blocks : float
        self.dataframe_declustered : np.ndarray

    self.get_extremes()
        self.extremes_method : str
        self.extremes_type : str
        self.threshold : float
        self.block_boundaries : np.ndarray
        self.extremes : pd.DataFrame
        self.extremes_rate : float
        self.plotting_position : str

    self.fit()
        self.distribution_name : str
        self.fit_method : str
        self.fit_parameters : tuple
        self.scipy_fit_options : dict
        self.sampler : emcee.EnsembleSampler
        self.mcmc_chain : np.ndarray
        self.fixed_parameters : np.ndarray

    self.generate_results()
        self.results : pd.DataFrame

    Private Attributes
    ------------------
    self.__init__()
        self.__status : dict

    Public Methods
    --------------
    self.to_pickle
    self.read_pickle
    self.get_extremes
    self.plot_extremes
    self.plot_mean_residual_life
    self.plot_parameter_stability
    self.test_extremes
    self.fit
    self.plot_trace
    self.plot_corner
    self.plot_posterior
    self.return_value
    self.confidence_interval
    self.generate_results
    self.plot_summary
    self.pdf
    self.cdf
    self.ppf
    self.isf
    self.plot_qq
    self.goodness_of_fit

    Private Methods
    ---------------
    self.__init__
    self.__get_blocks
    self.__update
    self.__repr__
    self.__get_return_period
    self.__run_mcmc
    self._kernel_fit_parameters
    self.__monte_carlo
    self.__delta
    self.__get_property
    """

    def __init__(self, dataframe, column=None, block_size=365.2425, gap_length=24):
        """
        Initializes the EVA class instance by taking a <dataframe> with values in <column> to analyze.
        Calculates number of blocks with <block_size>, accounting for gaps if <gap_length> is given.

        Parameters
        ----------
        dataframe : pd.DataFrame or pd.Series
            Pandas Dataframe or Series object containing data to be analyzed.
            Must have index array of type pd.DatetimeIndex.
        column : str or int, optional
            Name or index of column in <dataframe> with data to be analyzed.
            By default is <None> and takes first (0'th index) column from <dataframe>.
        block_size : float, optional
            Block size in days. Used to determine number of blocks in data (default=365.2425, one Gregorian year).
            Block size is used to estimate probabilities (return periods for observed data) for all methods
            and to extract extreme events in the 'Block Maxima' method.
            By default, it is one Gregorian year and results in return periods having units of years,
            i.e. a 100-<block_size> event by default is a 100-year return period event.
            Weekly would be <block_size=7> and monthly would be <block_size=365.2425/12>.
        gap_length : float, optional
            Gap length in hours. Gaps larger than <gap_length> are excluded when calculating total
            number of blocks of <block_size> in <dataframe>. Set to None to calculate number of blocks
            as "(last_date - first_date) / block_size". Default is 24 hours.
            It is also used in Block Maxima extreme value extraction method to get boundaries of blocks.
        """

        # Ensure passed <dataframe> is a pd.Dataframe object or can be converted to one
        if isinstance(dataframe, pd.DataFrame):
            self.dataframe = dataframe
        elif isinstance(dataframe, pd.Series):
            self.dataframe = dataframe.to_frame()
        else:
            raise TypeError(f'<dataframe> must be {pd.DataFrame} or {pd.Series}, {type(dataframe)} was passed')

        # Ensure <dataframe> index is pd.DatetimeIndex object
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            raise TypeError(f'<dataframe> index must be {pd.DatetimeIndex}, {type(dataframe.index)} was passed')
        self.dataframe.sort_index(ascending=True, inplace=True)

        # Ensure passed <column> represents a column within <dataframe>
        if column is not None:
            if isinstance(column, int):
                if column < len(self.dataframe.columns):
                    self.column = self.dataframe.columns[column]
                else:
                    raise ValueError(f'<column> with index {column} is not valid for '
                                     f'dataframe with {len(self.dataframe.columns)} columns')
            elif isinstance(column, str):
                if column in self.dataframe.columns:
                    self.column = column
                else:
                    raise ValueError(f'Column {column} is not valid for given dataframe.\n'
                                     f'Valid columns are {self.dataframe.columns}')
            else:
                raise TypeError(f'Column must be {str} or {int}, {type(column)} was passed.')
        else:
            self.column = self.dataframe.columns[0]

        # Ensure no nans are present in the <dataframe> <column>
        nancount = np.sum(np.isnan(self.dataframe[self.column].values))
        if nancount > 0:
            raise ValueError(f'<dataframe> contains {nancount} NaN values in column {self.column}.'
                             f'\nNaN values must be removed or filled before performing analysis.')

        # Ensure values in <dataframe> <column> are real numbers
        if not np.all(np.isreal(self.dataframe[self.column].values)):
            raise ValueError(f'Values in <dataframe> <column> must be real numbers,'
                             f' {self.dataframe[self.column].values.dtype} was passed')

        # Calculate number of blocks of <block_size> in <dataframe>
        self.block_size = block_size
        self.gap_length = gap_length
        self.number_of_blocks = self.__get_blocks(gap_length=self.gap_length)

        # Separate data into clusters using gap_length and plot each cluster independently
        # This way distant clusters are not connected on the plot
        if self.gap_length is not None:
            cluster_values = [[self.dataframe[self.column].values.copy()[0]]]
            cluster_indexes = [[self.dataframe.index.values.copy()[0]]]
            for index, value in zip(self.dataframe.index, self.dataframe[self.column].values):
                # New cluster encountered
                if index - cluster_indexes[-1][-1] > np.timedelta64(pd.Timedelta(hours=self.gap_length)):
                    cluster_values.append([value])
                    cluster_indexes.append([index])
                # Continuing within current cluster
                else:
                    cluster_values[-1].append(value)
                    cluster_indexes[-1].append(index)
            cluster_indexes = np.array(cluster_indexes)
            cluster_values = np.array(cluster_values)
            self.dataframe_declustered = np.array([cluster_indexes, cluster_values])
        else:
            self.dataframe_declustered = None

        # Initialize internal status
        # Internal status is used to delete calculation results when earlier methods are called
        # e.g. removes fit data and results when extreme events are exctracted. This prevents conflicts and errors
        self.__status = dict(
            extremes=False,
            fit=False,
            results=False
        )
        # Extremes extraction
        self.extremes_method = None
        self.extremes_type = None
        self.threshold = None
        self.block_boundaries = None
        self.extremes = None
        self.extremes_rate = None
        self.plotting_position = None
        # Extremes fit
        self.distribution_name = None
        self.fit_method = None
        self.fit_parameters = None
        self.scipy_fit_options = None
        self.sampler = None
        self.mcmc_chain = None
        self.fixed_parameters = None
        # Results
        self.results = None

    def __get_blocks(self, gap_length):
        """
        Calculates number of blocks of size <self.block_size> in <self.dataframe> <self.column>.

        Parameters
        ----------
        gap_length : float, optional
            Gap length in hours. Gaps larger than <gap_length> are excluded when calculating total
            number of blocks of <block_size> in <dataframe>. Set to None to calculate number of blocks
            as "(last_date - first_date) / block_size". Default is 24 hours.
            It is also used in Block Maxima extreme value extraction method to get boundaries of blocks.

        Returns
        -------
        n : float
            Number of blocks.
        """

        # Calculate number of blocks with gaps accounted for
        if gap_length is not None:
            timedelta = np.timedelta64(pd.Timedelta(hours=gap_length))
            # Eliminate gaps in data by shifting all values upstream of the gap downstream by <total_shift>
            new_index = self.dataframe.index.values.copy()
            for i in np.arange(1, len(new_index)):
                shift = new_index[i] - new_index[i-1]
                if shift > timedelta:
                    # Add 1/10 of gap_length to avoid duplicate dates
                    new_index[i:] -= shift - np.timedelta64(pd.Timedelta(hours=gap_length/10))
            series_range = np.float64(new_index[-1] - new_index[0])

        # Calculate number of blocks with gaps not accounted for
        else:
            series_range = np.float64((self.dataframe.index[-1] - self.dataframe.index[0]).value)

        return series_range / 1e9 / 60 / 60 / 24 / self.block_size

    def __update(self):
        """
        Updates internal state of the EVA class instance object.
        This method is used to delete calculation results when earlier methods are called.
        For example, removes all data related to fit and results when extreme events are extracted.
        """

        if not self.__status['extremes']:
            self.extremes_method = None
            self.extremes_type = None
            self.threshold = None
            self.block_boundaries = None
            self.extremes = None
            self.extremes_rate = None
            self.plotting_position = None

        if not self.__status['fit']:
            self.distribution_name = None
            self.fit_method = None
            self.fit_parameters = None
            self.scipy_fit_options = None
            self.sampler = None
            self.mcmc_chain = None
            self.fixed_parameters = None

        if not self.__status['results']:
            self.results = None

    def __repr__(self):
        """
        Generates a string with a summary of the EVA class instance object state.
        """

        series_range = (self.dataframe.index[-1] - self.dataframe.index[0]).value / 1e9 / 60 / 60 / 24

        summary = str(
            f'{" "*35}Extreme Value Analysis Summary\n'
            f'{"="*100}\n'
            f'Analyzed parameter{self.column:>29}{" "*6}Series length{series_range:29.2f} days\n'
            f'Gap length{self.gap_length:31.2f} hours{" "*6}'
            f'Adjusted series length{self.number_of_blocks*self.block_size:20.2f} days\n'
            f'Block size{self.block_size:32.2f} days{" "*6}Number of blocks{self.number_of_blocks:31.2f}\n'
            f'{"="*100}\n'
        )

        if self.__status['extremes']:
            summary += str(
                f'Number of extreme events{len(self.extremes):23}{" "*6}Extraction method{self.extremes_method:>30}\n'
                f'Extreme event rate{self.extremes_rate:16.2f} events/block{" "*6}'
                f'Plotting position{self.plotting_position:>30}\n'
                f'Threshold{self.threshold:38.2f}{" "*6}Extreme values type{self.extremes_type:>28}\n'
                f'{"="*100}\n'
            )
        else:
            summary += str(
                f'Number of extreme events{"N/A":>23}{" " * 6}Extraction method{"N/A":>30}\n'
                f'Extreme event rate{"N/A":>16} events/block{" " * 6}'
                f'Plotting position{"N/A":>30}\n'
                f'Threshold{"N/A":>38}{" "*6}Extreme values type{"N/A":>28}\n'
                f'{"=" * 100}\n'
            )

        if self.__status['fit']:
            if self.fit_method == 'MCMC':
                fit_parameters = self._kernel_fit_parameters(
                    burn_in=int(self.mcmc_chain.shape[1] / 2),
                    kernel_steps=100
                )
                summary += str(
                    f'Distribution{self.distribution_name:>35}{" " * 6}Fit method{"Markov chain Monte Carlo":>37}\n'
                    f'MCMC fit parameters (approximate){str(np.round(fit_parameters, 3)):>14}\n'
                    f'{"=" * 100}'
                )
            elif self.fit_method == 'MLE':
                summary += str(
                    f'Distribution{self.distribution_name:>35}{" " * 6}Fit method{"Maximum Likelihood Estimate":>37}\n'
                    f'MLE fit parameters{str(np.round(self.fit_parameters, 3)):>29}\n'
                    f'{"=" * 100}'
                )
        else:
            summary += str(
                f'Distribution{"N/A":>35}{" " * 6}Fit method{"N/A":>37}\n'
                f'Fit parameters{"N/A":>33}\n'
                f'{"=" * 100}'
            )

        return summary

    def to_pickle(self, path):
        """
        Exports EVA object to a .pyc file. Preserves all data and internal states.
        Can be used to save work, share analysis results, and to review work of others.

        Parameters
        ----------
        path : str
            Path to pickle file: e.g. <path:\to\pickle.pyc>.
        """

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def read_pickle(path):
        """
        Reads a .pyc file with EVA object. Loads all data and internal states.
        Can be used to save work, share analysis results, and to review work of others.

        Parameters
        ----------
        path : str
            Path to pickle file: e.g. <path:\to\pickle.pyc>.

        Returns
        -------
        file : EVA class instance object
            Saved EVA object with all data and internal state preserved.
        """

        with open(path, 'rb') as f:
            file = pickle.load(f)
        return file

    def get_extremes(self, method='BM', plotting_position='Weibull', extremes_type='high', **kwargs):
        """
        Extracts extreme values from <self.dataframe> <self.column> using the BM (Block Maxima)
        or the POT (Peaks Over Threshold) methods. If method is POT, also declusters extreme values using
        the runs method (aka minimum distance between independent events).

        Parameters
        ----------
        method : str, optional
            Peak extraction method. 'POT' for Peaks Over Threshold and 'BM' for Block Maxima (default='BM').
        plotting_position : str, optional
            Plotting position (default='Weibull'). Has no effect on return value inference,
            affects only some goodness of fit statistics and locations of observed extremes on the
            return values plot.
        extremes_type : str, optional
            Specifies type of extremes extracted: 'high' yields max values, 'low' yields min values (defaul='high').
            Use 'high' for extreme high values, use 'low' for extreme low values.
        kwargs
            for method='POT'
                threshold : float
                    Threshold for extreme value extraction.
                    Only values above (below, if <extremes_type='low'>) this threshold are extracted.
                r : float, optional
                    Minimum distance in hours between events for them to be considered independent.
                    Used to decluster extreme values using the runs method (default=24).
                adjust_threshold : bool, optional
                    If True, sets threshold equal to smallest/largest exceedance.
                    This way Generalized Pareto Distribution location parameter is strictly 0.
                    Eliminates instabilities associated with estimating location (default=True).

        Returns
        -------
        Creates a <self.extremes> dataframe with extreme values and return periods determined using
        the given plotting position as p=(rank-alpha)/(N+1-alpha-beta) and T=1/(1-p).
        """

        # Update internal status
        self.__status = dict(
            extremes=False,
            fit=False,
            results=False
        )
        self.__update()

        if extremes_type not in ['high', 'low']:
            raise ValueError(f'<extremes_type> must be high or low, {extremes_type} was passed')
        self.extremes_type = extremes_type

        # Block Maxima method
        if method == 'BM':

            assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

            # Set threshold to 0 for compatibility between BM and POT formulas
            self.extremes_method = 'Block Maxima'
            self.threshold = 0

            # Generate new index with gaps eliminated
            if self.gap_length is not None:
                gap_delta = np.timedelta64(pd.Timedelta(hours=self.gap_length))
                # Eliminate gaps in data by shifting all values upstream of the gap downstream by <total_shift>
                new_index = self.dataframe.index.values.copy()
                for i in np.arange(1, len(new_index)):
                    shift = new_index[i] - new_index[i-1]
                    if shift > gap_delta:
                        # Add 1/10 of gap_length to avoid duplicate dates
                        new_index[i:] -= shift - np.timedelta64(pd.Timedelta(hours=self.gap_length/10))
            else:
                new_index = self.dataframe.index.values.copy()

            # Create local reindexed dataframe with <new_index> and <id> column to get original datetime later
            local_dataframe = pd.DataFrame(
                data=self.dataframe[self.column].values.copy(),
                columns=[self.column], index=new_index
            )
            local_dataframe['id'] = np.arange(len(local_dataframe))

            # Find boundaries of blocks of <self.block_size>
            block_delta = np.timedelta64(pd.Timedelta(days=self.block_size))
            block_boundaries = [(new_index[0], new_index[0] + block_delta)]
            self.block_boundaries = [self.dataframe.index.values.copy()[0]]
            while block_boundaries[-1][-1] < local_dataframe.index.values[-1]:
                block_boundaries.append(
                    (block_boundaries[-1][-1], block_boundaries[-1][-1] + block_delta)
                )
                self.block_boundaries.append(
                    self.dataframe.index.values.copy()[
                        local_dataframe.truncate(before=block_boundaries[-1][0])['id'].values[0]
                    ]
                )
            self.block_boundaries.append(self.block_boundaries[-1] + block_delta)
            self.block_boundaries = np.array(self.block_boundaries)
            block_boundaries = np.array(block_boundaries)

            # Update number_of_blocks
            self.number_of_blocks = len(self.block_boundaries) - 1

            # Find extreme values within each block and associated datetime indexes from original dataframe
            extreme_values, extreme_indexes = [], []
            for i, block_boundary in enumerate(block_boundaries):
                if i == len(block_boundaries) - 1:
                    local_data = local_dataframe[local_dataframe.index >= block_boundary[0]]
                else:
                    local_data = local_dataframe[
                        (local_dataframe.index >= block_boundary[0]) & (local_dataframe.index < block_boundary[1])
                    ]
                if len(local_data) != 0:
                    if self.extremes_type == 'high':
                        extreme_values.append(local_data[self.column].values.copy().max())
                    else:
                        extreme_values.append(local_data[self.column].values.copy().min())
                    local_index = self.dataframe.index.values.copy()[
                        local_data[local_data[self.column].values == extreme_values[-1]]['id']
                    ]
                    if np.isscalar(local_index):
                        extreme_indexes.append(local_index)
                    else:
                        extreme_indexes.append(local_index[0])
            self.extremes = pd.DataFrame(data=extreme_values, columns=[self.column], index=extreme_indexes)

        # Peaks Over Threshold method
        elif method == 'POT':

            self.threshold = kwargs.pop('threshold')
            r = kwargs.pop('r', 24)
            adjust_threshold = kwargs.pop('adjust_threshold', True)
            assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

            self.extremes_method = 'Peaks Over Threshold'

            # Make sure correct number of blocks is used (overrides previously created BM values)
            if isinstance(self.number_of_blocks, int):
                self.number_of_blocks = self.__get_blocks(gap_length=self.gap_length)

            # Extract raw extremes
            if self.extremes_type == 'high':
                self.extremes = self.dataframe[self.dataframe[self.column] > self.threshold][self.column].to_frame()
            else:
                self.extremes = self.dataframe[self.dataframe[self.column] < self.threshold][self.column].to_frame()

            # Decluster raw extremes using runs method
            if r is not None:
                r = np.timedelta64(pd.Timedelta(hours=r))
                last_cluster_index = self.extremes.index.values.copy()[0]
                peak_cluster_values = [self.extremes[self.column].values.copy()[0]]
                peak_cluster_indexes = [self.extremes.index.values.copy()[0]]
                for index, value in zip(self.extremes.index, self.extremes[self.column].values):
                    # New cluster encountered
                    if index - last_cluster_index > r:
                        peak_cluster_values.append(value)
                        peak_cluster_indexes.append(index)
                    # Continuing within current cluster
                    else:
                        # Update cluster peak
                        if self.extremes_type == 'high':
                            if value > peak_cluster_values[-1]:
                                peak_cluster_values[-1] = value
                                peak_cluster_indexes[-1] = index
                        else:
                            if value < peak_cluster_values[-1]:
                                peak_cluster_values[-1] = value
                                peak_cluster_indexes[-1] = index
                    # Index of previous cluster - lags behind <index> by 1
                    last_cluster_index = index
                self.extremes = pd.DataFrame(
                    data=peak_cluster_values, index=peak_cluster_indexes, columns=[self.column]
                )

            # Update threshold to smallest/largest extreme value in order to fix the GPD location parameter at 0.
            # GPD is very unstable with non-zero location.
            if adjust_threshold:
                if self.extremes_type == 'high':
                    self.threshold = self.extremes[self.column].values.min()
                else:
                    self.threshold = self.extremes[self.column].values.max()

        else:
            raise ValueError(f'Method {method} not recognized')

        self.extremes.index.name = self.dataframe.index.name

        # Calculate rate of extreme events (events/block)
        self.extremes_rate = len(self.extremes) / self.number_of_blocks

        # Assign ranks to data with duplicate values having average of ranks they would have individually
        self.plotting_position = plotting_position
        self.extremes['Return Period'] = self.__get_return_period(plotting_position=self.plotting_position)

        # Update internal status
        self.__status = dict(
            extremes=True,
            fit=False,
            results=False
        )
        self.__update()

    def __get_return_period(self, plotting_position, return_cdf=False):
        """
        Assigns return periods to extracted extreme events and updates the <self.extremes> index.

        Parameters
        ----------
        plotting_position : str
            Plotting position. Has no effect on return value inference,
            affects only some goodness of fit statistics and locations of observed extremes on the
            return values plot.
        return_cdf : bool, optional
            If True, returns cdf of extracted extremes (default=False).
        """

        # Assign ranks to data with duplicate values having average of ranks they would have individually
        if self.extremes_type == 'high':
            ranks = scipy.stats.rankdata(self.extremes[self.column].values, method='average')
        else:
            ranks = len(self.extremes) + 1 - scipy.stats.rankdata(self.extremes[self.column].values, method='average')

        # Calculate return periods using a specified plotting position
        # https://matplotlib.org/mpl-probscale/tutorial/closer_look_at_plot_pos.html
        plotting_positions = {
            'ECDF': (0, 1),
            'Hazen': (0.5, 0.5),
            'Weibull': (0, 0),
            'Laplace': (-1, -1),
            'Tukey': (1 / 3, 1 / 3),
            'Blom': (3 / 8, 3 / 8),
            'Median': (0.3175, 0.3175),
            'Cunnane': (0.4, 0.4),
            'Gringorten': (0.44, 0.44),
            'Gumbel': (1, 1)
        }
        if plotting_position not in plotting_positions:
            raise ValueError(f'Plotting position {plotting_position} not recognized')
        alpha, beta = plotting_positions[plotting_position][0], plotting_positions[plotting_position][1]
        cdf = (ranks - alpha) / (len(self.extremes) + 1 - alpha - beta)
        if return_cdf:
            return cdf

        # Survival function - aka upper tail probability or probability of exceedance
        sf = 1 - cdf
        return 1 / sf / self.extremes_rate

    def plot_extremes(self):
        """
        Plots extracted extreme values on top of <self.dataframe> <self.column> observed time series.
        Shows boundaries of blocks for the Block Maxima method and threshold level for the Peaks Over Threshold method.

        Returns
        -------
        tuple(fig, ax)
        """

        # Make sure extreme values have been extracted
        if not self.__status['extremes']:
            raise RuntimeError('Extreme values have not been extracted. Run self.get_extremes() first')

        with plt.style.context('bmh'):
            fig, ax = plt.subplots(figsize=(12, 8))
            points = ax.scatter(
                self.extremes.index, self.extremes[self.column],
                edgecolors='white', marker='s', facecolors='k', s=40, lw=1, zorder=15
            )
            if self.gap_length is None:
                ax.plot(
                    self.dataframe.index, self.dataframe[self.column],
                    color='#3182bd', lw=.5, alpha=.8, zorder=5
                )
            else:
                for x, y in zip(self.dataframe_declustered[0], self.dataframe_declustered[1]):
                    ax.plot(x, y, color='#3182bd', lw=.5, alpha=.8, zorder=5)

            if self.extremes_method == 'Block Maxima':
                for _block in self.block_boundaries:
                    ax.axvline(_block, color='k', ls='--', lw=1, zorder=10)
            elif self.extremes_method == 'Peaks Over Threshold':
                ax.axhline(self.threshold, color='k', ls='--', lw=1, zorder=10)

            ax.set_title(f'Extreme Values Time Series, {self.extremes_method}')
            if len(self.dataframe.index.name) > 0:
                ax.set_xlabel(f'{self.dataframe.index.name}')
            else:
                ax.set_xlabel('Date')
            ax.set_ylabel(f'{self.column}')

            annot = ax.annotate(
                '', xy=(self.extremes.index[0], self.extremes[self.column].values[0]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='k', lw=1, zorder=25),
                zorder=30
            )
            point = ax.scatter(
                self.extremes.index[0], self.extremes[self.column].values[0],
                edgecolors='white', marker='s', facecolors='orangered', s=80, lw=1, zorder=20
            )
            point.set_visible(False)
            annot.set_visible(False)

            def update_annot(ind):
                n = ind['ind'][0]
                pos = points.get_offsets()[n]
                annot.xy = pos
                point.set_offsets(pos)
                text = str(
                    f'Date : {self.extremes.index[n]}\n'
                    f'Value : {self.extremes[self.column].values[n]:.2f}\n'
                    f'Return period : {self.extremes["Return Period"].values[n]:.2f}\n'
                    f'Plotting position : {self.plotting_position}'
                )
                annot.set_text(text)

            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    cont, ind = points.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        point.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            point.set_visible(False)
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect('motion_notify_event', hover)

            fig.tight_layout()

            return fig, ax

    def plot_mean_residual_life(self, thresholds=None, r=24, alpha=.95, extremes_type='high',
                                adjust_threshold=True, limit=10, plot=True):
        """
        Plots means of residuals against thresholds.
        Threshold should be chosen as the smallest threshold in a region where the mean residuals' plot
        is approximately linear. Generalized Pareto Distribution is asymptotically valid in this region.

        Parameters
        ----------
        thresholds : array_like, optional
            Array with threshold values for which the plot is generated.
            Default .95 quantile to max for 'high' and min to .05 quantile for 'low', 100 values.
        r : float, optional
            POT method only: minimum distance in hours between events for them to be considered independent.
            Used to decluster extreme values using the runs method (default=24).
        alpha : float, optional
            Confidence interval (default=.95). If None, doesn't plot or return confidence limits.
        extremes_type : str, optional
            Specifies type of extremes extracted: 'high' yields max values, 'low' yields min values (defaul='high').
            Use 'high' for extreme high values, use 'low' for extreme low values.
        adjust_threshold : bool, optional
            If True, sets threshold equal to smallest/largest exceedance.
            This way Generalized Pareto Distribution location parameter is strictly 0.
            Eliminates instabilities associated with estimating location (default=True).
        limit : int, optional
            Minimum number of exceedances (peaks) for which calculations are performed (default=10).
        plot : bool, optional
            Generates plot if True, returns data if False (default=True).

        Returns
        -------
        if plot=True (default) : tuple(fig, ax)
        if plot=False : tuple(thresholds, residuals, confidence_low, confidence_top)
        """

        if thresholds is None:
            if extremes_type == 'high':
                thresholds = np.linspace(
                    np.quantile(self.dataframe[self.column].values, .95),
                    self.dataframe[self.column].values.max(),
                    100
                )
            else:
                thresholds = np.linspace(
                    self.dataframe[self.column].values.min(),
                    np.quantile(self.dataframe[self.column].values, .05),
                    100
                )

        if np.isscalar(thresholds):
            raise ValueError('Thresholds must be an array. A scalar was provided')

        thresholds = np.sort(thresholds)
        if extremes_type == 'high':
            thresholds = thresholds[thresholds < self.dataframe[self.column].values.max()]
        else:
            thresholds = thresholds[thresholds > self.dataframe[self.column].values.min()]

        # Find mean residuals and 95% confidence interval for each threshold
        residuals, confidence = [], []
        true_thresholds = []
        for u in thresholds:
            self.get_extremes(
                method='POT', threshold=u, r=r,
                adjust_threshold=adjust_threshold, extremes_type=extremes_type
            )
            true_thresholds.append(self.threshold)
            exceedances = self.extremes[self.column].values - self.threshold
            # Flip exceedances around 0
            if extremes_type == 'low':
                exceedances *= -1
            if len(exceedances) > limit:
                residuals.append(exceedances.mean())
                # Ubiased estimator of sample variance of mean s^2/n
                confidence.append(
                    scipy.stats.norm.interval(
                        alpha=alpha, loc=exceedances.mean(),
                        scale=exceedances.std(ddof=1)/np.sqrt(len(exceedances))
                    )
                )
            else:
                residuals.append(np.nan)
                confidence.append((np.nan, np.nan))
        residuals = np.array(residuals)
        confidence = np.array(confidence)

        # Remove non-unique values
        if adjust_threshold:
            thresholds, mask = np.unique(true_thresholds, return_index=True)
            residuals = residuals[mask]
            confidence = confidence[mask]

        # Update internal status
        self.__status = dict(
            extremes=False,
            fit=False,
            results=False
        )
        self.__update()

        # Generate mean residual life plot
        if plot:
            with plt.style.context('bmh'):
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.set_title('Mean Residual Life Plot')
                ax.plot(thresholds, residuals, color='k', zorder=10, label='Mean residual life', lw=2)
                ax.plot(thresholds, confidence.T[0], ls='--', color='k', lw=0.5, zorder=10)
                ax.plot(thresholds, confidence.T[1], ls='--', color='k', lw=0.5, zorder=10)
                ax.fill_between(
                    thresholds, confidence.T[0], confidence.T[1],
                    alpha=.1, color='k', label=f'{alpha*100:.0f}% confidence interval', zorder=5
                )
                ax.legend()
                ax.set_xlabel('Threshold')
                ax.set_ylabel('Mean Residual')
                fig.tight_layout()
                return fig, ax
        else:
            return thresholds, residuals, confidence.T[0], confidence.T[1]

    def plot_parameter_stability(self, thresholds=None, r=24, alpha=.95, extremes_type='high',
                                 adjust_threshold=True, limit=10, plot=True, dx='1e-10', precision=100):
        """
        Plots shape and modified scale paramters of the Generalized Pareto Distribution (GPD) against thresholds.
        GPD is asymptotically valid in a region where these parameters are approximately linear.

        Parameters
        ----------
        thresholds : array_like, optional
            Array with threshold values for which the plot is generated.
            Default .95 quantile to max for 'high' and min to .05 quantile for 'low', 100 values.
        r : float, optional
            Minimum distance in hours between events for them to be considered independent.
            Used to decluster extreme values using the runs method (default=24).
        alpha : float, optional
            Confidence interval (default=.95). If None, doesn't plot or return confidence limits.
        extremes_type : str, optional
            Specifies type of extremes extracted: 'high' yields max values, 'low' yields min values (defaul='high').
            Use 'high' for extreme high values, use 'low' for extreme low values.
        adjust_threshold : bool, optional
            If True, sets threshold equal to smallest/largest exceedance.
            This way Generalized Pareto Distribution location parameter is strictly 0.
            Eliminates instabilities associated with estimating location (default=True).
        limit : int, optional
            Minimum number of exceedances (peaks) for which calculations are performed (default=10).
        plot : bool, optional
            Generates plot if True, returns data if False (default=True).
        dx : str, optional
            String representing a float, which represents spacing at which partial derivatives
            are estimated (default='1e-10').
        precision : int, optional
            Precision of floating point calculations (see mpmath library documentation) (default=100).
            Derivative estimated with low <precision> value may have
            a significant error due to rounding and under-/overflow.

        Returns
        -------
        if plot=True (default) : tuple(fig, ax)
        if plot=False :
            if alpha is None : tuple(thresholds, shapes, modified_scales)
            if alpha is passed : tuple(thresholds, shapes, modified_scales, shapes_confidence, scales_confidence)
        """

        if thresholds is None:
            if extremes_type == 'high':
                thresholds = np.linspace(
                    np.quantile(self.dataframe[self.column].values, .95),
                    self.dataframe[self.column].values.max(),
                    100
                )
            else:
                thresholds = np.linspace(
                    self.dataframe[self.column].values.min(),
                    np.quantile(self.dataframe[self.column].values, .05),
                    100
                )

        if np.isscalar(thresholds):
            raise ValueError('Thresholds must be an array. A scalar was provided')

        thresholds = np.sort(thresholds)
        if extremes_type == 'high':
            thresholds = thresholds[thresholds < self.dataframe[self.column].values.max()]
        else:
            thresholds = thresholds[thresholds > self.dataframe[self.column].values.min()]

        shapes, modified_scales = [], []
        shapes_confidence, scales_confidence = [], []
        true_thresholds = []
        for u in thresholds:
            self.get_extremes(
                method='POT', threshold=u, r=r,
                adjust_threshold=adjust_threshold, extremes_type=extremes_type
            )
            true_thresholds.append(self.threshold)
            exceedances = self.extremes[self.column].values - self.threshold
            # Flip exceedances around 0
            if extremes_type == 'low':
                exceedances *= -1

            if len(exceedances) > limit:
                shape, loc, scale = scipy.stats.genpareto.fit(exceedances, floc=0)
                shapes.append(shape)

                # Define modified scale function (used as scalar function for delta method)
                if extremes_type == 'high':
                    def mod_scale_function(*theta):
                        return theta[1] - theta[0] * true_thresholds[-1]
                else:
                    def mod_scale_function(*theta):
                        return theta[1] + theta[0] * true_thresholds[-1]

                modified_scales.append(mod_scale_function(shape, scale))

                if alpha is not None:
                    with mpmath.workdps(precision):
                        # Define modified log_likehood function
                        def log_likelihood(*theta):
                            return mpmath.fsum(
                                [
                                    mpmath.log(
                                        coastlib.stats.distributions.genpareto.pdf(
                                            x=_x, shape=theta[0], loc=0, scale=theta[1]
                                        )
                                    ) for _x in exceedances
                                ]
                            )

                        # Calculate delta (gradient) of scalar_function
                        if extremes_type == 'high':
                            delta_scalar = np.array(
                                [
                                    [-true_thresholds[-1]],
                                    [1]
                                ]
                            )
                        else:
                            delta_scalar = np.array(
                                [
                                    [true_thresholds[-1]],
                                    [1]
                                ]
                            )

                        # Calculate observed information matrix (negative hessian of log_likelihood)
                        observed_information = -coastlib.math.derivatives.hessian(
                            func=log_likelihood, n=2, coordinates=[shape, scale], dx=dx, precision=precision
                            ).astype(np.float64)
                        covariance = np.linalg.inv(observed_information)

                    # Estimate modified scale parameter confidence interval using delta method
                    variance = np.dot(
                        np.dot(delta_scalar.T, covariance), delta_scalar
                    ).flatten()[0]
                    scales_confidence.append(
                        scipy.stats.norm.interval(
                            alpha=alpha, loc=modified_scales[-1], scale=np.sqrt(variance)
                        )
                    )

                    # Estimate shape parameter confidence interval directly from covariance matrix
                    shapes_confidence.append(
                        scipy.stats.norm.interval(
                            alpha=alpha, loc=shape, scale=np.sqrt(covariance[0][0])
                        )
                    )
            # Number of exceedances below the limit
            else:
                shapes.append(np.nan)
                modified_scales.append(np.nan)
                if alpha is not None:
                    shapes_confidence.append((np.nan, np.nan))
                    scales_confidence.append((np.nan, np.nan))

        # Convert results to np.ndarray objects
        shapes = np.array(shapes)
        modified_scales = np.array(modified_scales)
        if alpha is not None:
            shapes_confidence = np.array(shapes_confidence)
            scales_confidence = np.array(scales_confidence)

        # Remove non-unique values
        if adjust_threshold:
            thresholds, mask = np.unique(true_thresholds, return_index=True)
            shapes = shapes[mask]
            modified_scales = modified_scales[mask]
            if alpha is not None:
                shapes_confidence = shapes_confidence[mask]
                scales_confidence = scales_confidence[mask]

        # Update internal status
        self.__status = dict(
            extremes=False,
            fit=False,
            results=False
        )
        self.__update()

        if plot:
            with plt.style.context('bmh'):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex='all')
                ax1.set_title('Parameter Stability Plot')
                ax1.plot(thresholds, shapes, color='k', zorder=10, label='Shape parameter')
                ax2.plot(thresholds, modified_scales, color='k', zorder=10, label='Modified scale parameter', lw=2)

                if alpha is not None:
                    ax1.plot(thresholds, shapes_confidence.T[0], ls='--', color='k', lw=0.5)
                    ax1.plot(thresholds, shapes_confidence.T[1], ls='--', color='k', lw=0.5)
                    ax2.plot(thresholds, scales_confidence.T[0], ls='--', color='k', lw=0.5)
                    ax2.plot(thresholds, scales_confidence.T[1], ls='--', color='k', lw=0.5)

                    ax1.fill_between(
                        thresholds, shapes_confidence.T[0], shapes_confidence.T[1],
                        alpha=.1, color='k', label=f'{alpha*100:.0f}% confidence interval'
                    )
                    ax2.fill_between(
                        thresholds, scales_confidence.T[0], scales_confidence.T[1],
                        alpha=.1, color='k', label=f'{alpha*100:.0f}% confidence interval'
                    )

                ax2.set_xlabel('Threshold')
                ax1.set_ylabel('Shape parameter')
                ax2.set_ylabel('Modified scale parameter')

                ax1.legend()
                ax2.legend()
                fig.tight_layout()
                return fig, (ax1, ax2)
        else:
            if alpha is None:
                return thresholds, shapes, modified_scales
            else:
                return thresholds, shapes, modified_scales, shapes_confidence, scales_confidence

    def test_extremes(self, method, **kwargs):
        """
        Provides multiple methods to test independece of extracted extreme values.

        Parameters
        ----------
        method : str
            Method for testing extreme values' independence.
            Accepted methods:
                'autocorrelation' - generates an autocorrelation plot
                    http://www.statsmodels.org/stable/generated/
                    statsmodels.tsa.stattools.acf.html#statsmodels.tsa.stattools.acf
                'lag plot' - generates a lag plot for a given lag
                'runs test' - return runs test statistic
                    https://en.wikipedia.org/wiki/Wald%E2%80%93Wolfowitz_runs_test
        kwargs
            for autocorrelation:
                plot : bool, optional
                    Generates plot if True, returns data if False (default=True).
                nlags : int, optional
                    Number of lags to return autocorrelation for (default for all possible lags).
                alpha : float, optional
                    Confidence interval (default=.95). If None, doesn't plot or return confidence limits.
                unbiased : bool, optional
                    If True, then denominators for autocovariance are n-k, otherwise n (default=False)
            for lag plot:
                plot : bool, optional
                    Generates plot if True, returns data if False (default=True).
                lag : int, optional
                    Lag value (default=1).
            for runs test:
                alpha : float, optional
                    Significance level (default=0.05).

        Returns
        -------
        for autocorrelation:
            if plot=True : tuple(fig, ax)
            if plot=False : tuple(lags, acorr, ci_low, ci_top)
        for lag plot:
            if plot=True : tuple(fig, ax)
            if plot=False : tuple(x, y)
        for runs test:
            str(test summary)
        """

        if not self.__status['extremes']:
            raise RuntimeError('Extreme values have not been extracted. Nothing to test')

        if method == 'autocorrelation':
            plot = kwargs.pop('plot', True)
            nlags = kwargs.pop('nlags', len(self.extremes) - 1)
            alpha = kwargs.pop('alpha', .95)
            unbiased = kwargs.pop('unbiased', False)
            assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

            acorr, ci = sm.tsa.stattools.acf(
                x=self.extremes[self.column].values, alpha=1-alpha, nlags=nlags, unbiased=unbiased
            )
            ci_low, ci_top = ci.T[0] - acorr, ci.T[1] - acorr
            if plot:
                with plt.style.context('bmh'):
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.vlines(np.arange(nlags+1), [0], acorr, lw=1, color='k', zorder=15)
                    points = ax.scatter(
                        np.arange(nlags+1), acorr, marker='o', s=40, lw=1,
                        facecolor='k', edgecolors='white', zorder=20, label='Autocorrelation value'
                    )
                    ax.plot(np.arange(nlags+1)[1:], ci_low[1:], color='k', lw=.5, ls='--', zorder=15)
                    ax.plot(np.arange(nlags+1)[1:], ci_top[1:], color='k', lw=.5, ls='--', zorder=15)
                    ax.fill_between(
                        np.arange(nlags+1)[1:], ci_low[1:], ci_top[1:],
                        color='k', alpha=.1, zorder=5, label=f'{alpha*100:.0f}% confidence interval'
                    )
                    ax.axhline(0, color='k', lw=1, ls='--', zorder=10)
                    ax.legend()
                    ax.set_title('Autocorrelation plot')
                    ax.set_xlabel('Lag')
                    ax.set_ylabel('Correlation coefficient')

                    annot = ax.annotate(
                        '', xy=(0, 0),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='k', lw=1, zorder=25),
                        zorder=30
                    )
                    point = ax.scatter(
                        0, 0,
                        edgecolors='white', marker='o', facecolors='orangered', s=80, lw=1, zorder=25
                    )
                    point.set_visible(False)
                    annot.set_visible(False)

                    def update_annot(ind):
                        n = ind['ind'][0]
                        pos = points.get_offsets()[n]
                        annot.xy = pos
                        point.set_offsets(pos)
                        text = str(
                            f'Lag : {np.arange(nlags+1)[n]:d}\n'
                            f'Correlation : {acorr[n]:.2f}'
                        )
                        annot.set_text(text)

                    def hover(event):
                        vis = annot.get_visible()
                        if event.inaxes == ax:
                            cont, ind = points.contains(event)
                            if cont:
                                update_annot(ind)
                                annot.set_visible(True)
                                point.set_visible(True)
                                fig.canvas.draw_idle()
                            else:
                                if vis:
                                    annot.set_visible(False)
                                    point.set_visible(False)
                                    fig.canvas.draw_idle()

                    fig.canvas.mpl_connect('motion_notify_event', hover)

                    fig.tight_layout()

                    return fig, ax

            else:
                return np.arange(nlags+1), acorr, ci_low, ci_top

        elif method == 'lag plot':
            plot = kwargs.pop('plot', True)
            lag = kwargs.pop('lag', 1)
            assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

            if lag == 0:
                x = self.extremes[self.column].values
            else:
                x = self.extremes[self.column].values[:-lag]
            y = self.extremes[self.column].values[lag:]

            if plot:
                with plt.style.context('bmh'):
                    fig, ax = plt.subplots(figsize=(12, 8))
                    points = ax.scatter(
                        x, y, marker='o', facecolor='k', s=40, edgecolors='white', lw=1, zorder=5
                    )
                    ax.set_xlabel(f'{self.column} i')
                    ax.set_ylabel(f'{self.column} i+{lag}')
                    ax.set_title('Extreme Values Lag Plot')

                    annotation = ax.annotate(
                        "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='k', lw=1, zorder=25),
                        zorder=30
                    )
                    point = ax.scatter(
                        np.nanmean(x), np.nanmean(y),
                        edgecolors='white', marker='o', facecolors='orangered', s=80, lw=1, zorder=20
                    )
                    point.set_visible(False)
                    annotation.set_visible(False)

                    def update_annotation(ind):
                        pos = points.get_offsets()[ind['ind'][0]]
                        annotation.xy = pos
                        point.set_offsets(pos)
                        text = "{}".format(" ".join(
                            [
                                f'{self.extremes.index[n]} : {ind["ind"][0]}\n'
                                f'{self.extremes.index[n+lag]} : {ind["ind"][0]+lag}'
                                for n in ind['ind']
                            ]))
                        annotation.set_text(text)

                    def hover(event):
                        vis = annotation.get_visible()
                        if event.inaxes == ax:
                            cont, ind = points.contains(event)
                            if cont:
                                update_annotation(ind)
                                annotation.set_visible(True)
                                point.set_visible(True)
                                fig.canvas.draw_idle()
                            else:
                                if vis:
                                    annotation.set_visible(False)
                                    point.set_visible(False)
                                    fig.canvas.draw_idle()

                    fig.canvas.mpl_connect('motion_notify_event', hover)

                    fig.tight_layout()

                    return fig, ax
            else:
                return x, y

        elif method == 'runs test':
            alpha = kwargs.pop('alpha', .05)
            assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

            # Calculate number of runs of shifted series
            s = self.extremes[self.column].values - np.quantile(self.extremes[self.column].values, .5)
            n_plus = np.sum(s > 0)
            n_minus = np.sum(s < 0)
            n_runs = 1
            for i in range(1, len(s)):
                # Change of sign
                if s[i] * s[i-1] < 0:
                    n_runs += 1
            mean = 2 * n_plus * n_minus / len(s) + 1
            variance = (mean - 1) * (mean - 2) / (len(s) - 1)
            test_statistic = (n_runs-mean)/np.sqrt(variance)

            return str(
                f'Ho : data is random\n'
                f'Ha : data is not random\n\n'
                f'Test statistic : N = {test_statistic:.2f}\n'
                f'Significanse level : alpha = {alpha}\n'
                f'Critical value : Nalpha = {scipy.stats.norm.ppf(1 - alpha / 2):.2f}\n'
                f'Reject Ho if |N| > Nalpha'
            )

        else:
            raise ValueError(f'Method {method} not recognized. Try: autocorrelation')

    def fit(self, distribution_name, fit_method='MLE', **kwargs):
        """
        Depending on fit method, either creates a tuple with maximum likelihood estimate (MLE)
        or an array with samples drawn from posterior distribution of parameters (MCMC).

        Parameters
        ----------
        distribution_name : str
            Scipy distribution name (see https://docs.scipy.org/doc/scipy/reference/stats.html).
        fit_method : str, optional
            Fit method - MLE (Maximum Likelihood Estimate, scipy)
            or Markov chain Monte Carlo (MCMC, emcee) (default='MLE').
        kwargs:
            for MLE:
                scipy_fit_options : dict, optional
                    Special scipy fit options like <fc>, <loc>, or <floc>.
                    For GPD scipy_fit_options=dict(floc=0) by default (fixed location parameter at 0).
                    This parameter is carried over to further calculations, such as confidence interval.
            for MCMC:
                nsamples : int, optional
                    Number of samples each walker draws (default=1000).
                    Larger values result in longer processing time, but can lead to better convergence.
                nwalkers : int, optional
                    Number of walkers (default=200). Each walker explores the parameter space.
                    Larger values result in longer processing time,
                    but more parameter space is explored (higher chance to escape local maxima).
                log_prior : callable, optional
                    Function taking one parameter - list with fit parameters (theta).
                    Returns sum of log-probabilities (logpdf) for each parameter within theta.
                    By default is uniform for each parameter.
                    read http://dfm.io/emcee/current/user/line/
                    Default functions are defined only for 3-parameter GEV and 3- and 2-parameter (loc=0) GPD.
                log_likelihood : callable, optional
                    Function taking one parameter - list with fit parameters (theta).
                    Returns log-likelihood (sum of logpdf) for given parameters.
                    By default is sum(logpdf) of scipy distribution with <distribution_name>.
                    read http://dfm.io/emcee/current/user/line/
                    Default functions are defined only for 3-parameter GEV and 3- and 2-parameter (loc=0) GPD.
                starting_bubble : float, optional
                    Radius of bubble from <starting_position> within which
                    starting parameters for each walker are set (default=1e-2).
                starting_position : array_like, optional
                    Array with starting parameters for each walker (default=None).
                    If None, then zeroes are chosen as starting parameter.
                fixed_parameters : array_like, optional
                    An array with tuples with index of parameter being fixed "i" and parameter value "v" [(i, v),...]
                    for each parameter being fixed (default [(1,0)] for GPD, None for other).
                    Works only with custom distributions. Must be sorted in ascending order by "i".
        """

        # Make sure extreme values have been extracted
        if not self.__status['extremes']:
            raise RuntimeError('Extreme values have not been extracted. Nothing to fit')

        # Update internal status
        self.__status = dict(
            extremes=True,
            fit=False,
            results=False
        )
        self.__update()

        if fit_method == 'MLE':

            if distribution_name == 'genpareto':
                self.scipy_fit_options = kwargs.pop('scipy_fit_options', dict(floc=0))
            else:
                self.scipy_fit_options = kwargs.pop('scipy_fit_options', {})
            assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

            # Create local distribution object
            distribution_object = getattr(scipy.stats, distribution_name)

            exceedances = self.extremes[self.column].values - self.threshold
            # Flip exceedances around 0
            if self.extremes_type == 'low':
                exceedances *= -1

            self.fit_parameters = distribution_object.fit(exceedances, **self.scipy_fit_options)

        elif fit_method == 'MCMC':

            self.mcmc_chain = self.__run_mcmc(distribution_name, **kwargs)

        else:
            raise ValueError(f'Fit method {fit_method} not recognized')

        # On successful fit assign the fit_ variables
        self.fit_method = fit_method
        self.distribution_name = distribution_name

        # Update internal status
        self.__status = dict(
            extremes=True,
            fit=True,
            results=False
        )
        self.__update()

    def __run_mcmc(self, distribution_name, nsamples=1000, nwalkers=200, **kwargs):
        """
        Runs emcee Ensemble Sampler to sample posteriot probability of fit parameters given observed data.
        Returns sampler chain with <nsamples> for each parameter for each <nwalkers>.
        See http://dfm.io/emcee/current/

        Parameters
        ----------
        distribution_name : str
            Scipy distribution name (see https://docs.scipy.org/doc/scipy/reference/stats.html).
        nsamples : int, optional
            Number of samples each walker draws (default=1000).
            Larger values result in longer processing time, but can lead to better convergence.
        nwalkers : int, optional
            Number of walkers (default=200). Each walker explores the parameter space.
            Larger values result in longer processing time,
            but more parameter space is explored (higher chance to escape local maxima).
        kwargs
            log_prior : callable, optional
                Function taking one parameter - list with fit parameters (theta).
                Returns sum of log-probabilities (logpdf) for each parameter within theta.
                By default is uniform for each parameter.
                read http://dfm.io/emcee/current/user/line/
                Default functions are defined only for 3-parameter GEV and 3- and 2-parameter (loc=0) GPD.
            log_likelihood : callable, optional
                Function taking one parameter - list with fit parameters (theta).
                Returns log-likelihood (sum of logpdf) for given parameters.
                By default is sum(logpdf) of scipy distribution with <distribution_name>.
                read http://dfm.io/emcee/current/user/line/
                Default functions are defined only for 3-parameter GEV and 3- and 2-parameter (loc=0) GPD.
            starting_bubble : float, optional
                Radius of bubble from <starting_position> within which
                starting parameters for each walker are set (default=1e-2).
            starting_position : array_like, optional
                Array with starting parameters for each walker (default=None).
                If None, then zeroes are chosen as starting parameter.
            fixed_parameters : array_like, optional
                An array with tuples with index of parameter being fixed "i" and parameter value "v" [(i, v),...]
                for each parameter being fixed (default [(1,0)] for GPD, None for other).
                Works only with custom distributions. Must be sorted in ascending order by "i".

        Returns
        -------
        Generates an np.ndarray in self.mcmc_chain
            Ensemble Sampler chain with <nsamples> for each parameter for each <nwalkers>.
        """

        log_prior = kwargs.pop('log_prior', None)
        log_likelihood = kwargs.pop('log_likelihood', None)
        starting_bubble = kwargs.pop('starting_bubble', 1e-2)
        starting_position = kwargs.pop('starting_position', None)
        if distribution_name == 'genpareto':
            self.fixed_parameters = kwargs.pop('fixed_parameters', [(1, 0)])
        else:
            self.fixed_parameters = kwargs.pop('fixed_parameters', None)
        assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

        if self.fixed_parameters == [(1, 0)] and distribution_name == 'genpareto':
            pass
        else:
            if self.fixed_parameters is not None:
                if (log_prior is None) or (log_likelihood is None) or (starting_position is None):
                    raise ValueError(
                        '<fixed_parameter> only works with custom prior and likelihood functions.\n'
                        'Starting position should be provided for the fixed_parameters case'
                    )

        distribution_object = getattr(scipy.stats, distribution_name)
        exceedances = self.extremes[self.column].values - self.threshold
        # Flip exceedances around 0
        if self.extremes_type == 'low':
            exceedances *= -1

        # Define log_prior probability function (uniform by default)
        if log_prior is None:
            if distribution_name == 'genpareto':
                # https://en.wikipedia.org/wiki/Generalized_Pareto_distribution
                if self.fixed_parameters == [(1, 0)]:
                    def log_prior(theta):
                        shape, scale = theta
                        if scale <= 0:
                            return -np.inf
                        return 0
                else:
                    def log_prior(theta):
                        shape, loc, scale = theta
                        # Parameter constraint
                        if scale <= 0:
                            return -np.inf
                        # Support constraint
                        if shape >= 0:
                            condition = np.all(exceedances >= loc)
                        else:
                            condition = np.all(exceedances >= loc) and np.all(exceedances <= loc - scale / shape)
                        if condition:
                            return 0
                        else:
                            return -np.inf
            elif distribution_name == 'genextreme':
                # https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
                def log_prior(theta):
                    shape, loc, scale = theta
                    # Parameter constraint
                    if scale <= 0:
                        return -np.inf
                    # Support constraint (scipy shape has inverted sign)
                    shape *= -1
                    if shape > 0:
                        condition = np.all(exceedances >= loc - scale / shape)
                    elif shape == 0:
                        condition = True
                    else:
                        condition = np.all(exceedances <= loc - scale / shape)
                    if condition:
                        return 0
                    else:
                        return -np.inf
            else:
                raise NotImplementedError(
                    f'Log-prior function is not implemented for {distribution_name} parameters.\n'
                    f'Define manually and pass to <log_prior=>.'
                )

        # Define log_likelihood function
        if log_likelihood is None:
            if distribution_name == 'genpareto':
                # https://en.wikipedia.org/wiki/Generalized_Pareto_distribution
                if self.fixed_parameters == [(1, 0)]:
                    def log_likelihood(theta):
                        shape, scale = theta
                        if scale <= 0:
                            return -np.inf
                        return np.sum(distribution_object.logpdf(exceedances, shape, 0, scale))
                else:
                    def log_likelihood(theta):
                        shape, loc, scale = theta
                        # Parameter constraint
                        if scale <= 0:
                            return -np.inf
                        # Support constraint
                        if shape >= 0:
                            condition = np.all(exceedances >= loc)
                        else:
                            condition = np.all(exceedances >= loc) and np.all(exceedances <= loc - scale / shape)
                        if condition:
                            return np.sum(distribution_object.logpdf(exceedances, *theta))
                        else:
                            return -np.inf
            elif distribution_name == 'genextreme':
                # https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
                def log_likelihood(theta):
                    shape, loc, scale = theta
                    # Parameter constraint
                    if scale <= 0:
                        return -np.inf
                    # Support constraint (scipy shape has inverted sign)
                    shape *= -1
                    if shape > 0:
                        condition = np.all(exceedances >= loc - scale / shape)
                    elif shape == 0:
                        condition = True
                    else:
                        condition = np.all(exceedances <= loc - scale / shape)
                    if condition:
                        return np.sum(distribution_object.logpdf(exceedances, *theta))
                    else:
                        return -np.inf
            else:
                raise NotImplementedError(
                    f'Log-likelihood function is not implemented for {distribution_name} parameters.\n'
                    f'Define manually and pass to <log_likelihood=>.'
                )

        # Define log_posterior probability function (not exact - excludes marginal evidence probability)
        def log_posterior(theta):
            return log_likelihood(theta) + log_prior(theta)

        # Set MCMC walkers' starting positions to 0
        # (setting to MLE makes algorithm unstable due to being stuck in local maxima)
        if starting_position is None:
            if distribution_name == 'genpareto' and self.fixed_parameters == [(1, 0)]:
                theta_0 = np.array([0, 0])
            elif distribution_name in ['genextreme', 'genpareto']:
                theta_0 = np.array([0, 0, 0])
            else:
                theta_0 = distribution_object.fit(exceedances)
            starting_position = [[0] * len(theta_0) for _ in range(nwalkers)]

        # Randomize starting positions to force walkers explore the parameter space
        starting_position = [
            np.array(sp) + starting_bubble * np.random.randn(len(starting_position[0]))
            for sp in starting_position
        ]
        if len(starting_position) != nwalkers:
            raise ValueError(f'Number of starting positions {len(starting_position)} '
                             f'must be equal to number of walkers {nwalkers}')
        ndim = len(starting_position[0])

        # Setup the Ensemble Sampler and draw samples from posterior distribution for specified number of walkers
        self.__sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
        self.__sampler.run_mcmc(starting_position, nsamples)

        # Fill in fixed parameter values
        sampler_chain = self._EVA__sampler.chain.copy()
        if self.fixed_parameters is not None:
            fp = np.transpose(self.fixed_parameters)
            ndim = sampler_chain.shape[-1] + len(self.fixed_parameters)
            mcmc_chain = np.array(
                [
                    [
                        [np.nan] * ndim for _ in range(sampler_chain.shape[1])
                    ] for _ in range(sampler_chain.shape[0])
                ]
            )
            for i in range(mcmc_chain.shape[0]):
                for j in range(mcmc_chain.shape[1]):
                    counter = 0
                    for k in range(mcmc_chain.shape[2]):
                        if k in fp[0]:
                            mcmc_chain[i][j][k] = fp[1][fp[0] == k][0]
                        else:
                            mcmc_chain[i][j][k] = sampler_chain[i][j][counter]
                            counter += 1
            sampler_chain = np.array(mcmc_chain)

        return sampler_chain

    def _kernel_fit_parameters(self, burn_in, kernel_steps=1000):
        """
        Estimate mode of each parameter as peaks of gaussian kernel.

        Parameters
        ----------
        burn_in : int
            Number of samples to discard. Samples, before the series converges, should be discarded.
        kernel_steps : int, optional
            Number of bins (kernel support points) to determine mode (default=1000).

        Returns
        -------
        np.ndarray
            Modes of parameters.
        """

        if not self.__status['fit']:
            raise ValueError('No fit information found. Run self.fit() method first')

        if self.fit_method != 'MCMC':
            raise ValueError('Fit method must be MCMC')

        # Load samples
        ndim = self.mcmc_chain.shape[-1]
        samples = self.mcmc_chain[:, burn_in:, :].reshape((-1, ndim))

        # Estimate mode of each parameter as peaks of gaussian kernel.
        parameters = []
        for i, p in enumerate(samples.T):
            if self.fixed_parameters is None or (i not in np.transpose(self.fixed_parameters)[0]):
                p_filtered = p[~np.isnan(p)]
                kernel = scipy.stats.gaussian_kde(p_filtered)
                support = np.linspace(
                    np.quantile(p_filtered, .1), np.quantile(p_filtered, .9),
                    kernel_steps
                )
                density = kernel.evaluate(support)
                parameters.append(support[density.argmax()])
            else:
                parameters.append(p[0])

        return np.array(parameters)

    def plot_trace(self, burn_in, true_theta=None, labels=None):
        """
        Plots traces for each parameter. Each trace plot shows all samples for each walker
        after first <burn_in> samples are discarded. This method is used to verify fit stability
        and to determine the optimal <burn_in> value.

        Parameters
        ----------
        burn_in : int
            Number of samples to discard. Samples, before the series converges, should be discarded.
        true_theta : array_like, optional
            Array with true (known) values of parameters (default=None). If given, are shown on trace plots.
        labels : array_like, optional
            List of labels for each parameter (e.g. shape, loc, scale) (default - index).

        Returns
        -------
        tuple(fig, axes)
        """

        # Make sure self.mcmc_chain exists
        if self.mcmc_chain is None:
            raise RuntimeError('No mcmc_chain attribute found.')

        if labels is None:
            labels = [f'Parameter {i+1}' for i in range(self.__sampler.chain.shape[-1])]

        # Generate trace plot
        ndim = self.__sampler.chain.shape[-1]
        with plt.style.context('bmh'):
            fig, axes = plt.subplots(ndim, 1, figsize=(12, 8), sharex='all')
            if ndim == 1:
                axes.set_title('MCMC Trace Plot')
                axes.set_xlabel('Sample number')
            else:
                axes[0].set_title('MCMC Trace Plot')
                axes[-1].set_xlabel('Sample number')
            for i in range(ndim):
                for swalker in self.__sampler.chain:
                    if ndim == 1:
                        axes.plot(
                            np.arange(len(swalker.T[i]))[burn_in:],
                            swalker.T[i][burn_in:],
                            color='k', lw=0.1, zorder=5
                        )
                        axes.set_ylabel(labels[i])
                    else:
                        axes[i].plot(
                            np.arange(len(swalker.T[i]))[burn_in:],
                            swalker.T[i][burn_in:],
                            color='k', lw=0.1, zorder=5
                        )
                        axes[i].set_ylabel(labels[i])
                if true_theta is not None:
                    if ndim == 1:
                        axes.axhline(true_theta[i], color='orangered', lw=2, zorder=10)
                    else:
                        axes[i].axhline(true_theta[i], color='orangered', lw=2, zorder=10)
            fig.tight_layout()
        return fig, axes

    def plot_corner(self, burn_in, bins=100, labels=None, figsize=(12, 12), **kwargs):
        """
        Generate corner plot showing the projections of a data set in a multi-dimensional space.
        See https://corner.readthedocs.io/en/latest/api.html#corner.corner

        Parameters
        ----------
        burn_in : int
            Number of samples to discard. Samples, before the series converges, should be discarded.
        bins : int, optional
            See https://corner.readthedocs.io/en/latest/api.html#corner.corner (default=50).
        labels : array_like, optional
            List of labels for each parameter (e.g. shape, loc, scale) (default - index).
        figsize : tuple, optional
            Figure size (default=(12, 12)).
        kwargs
            Corner plot keywords. See https://corner.readthedocs.io/en/latest/api.html#corner.corner

        Returns
        -------
        tuple(fig, ax)
        """

        # Make sure self.mcmc_chain exists
        if self.mcmc_chain is None:
            raise RuntimeError('mcmc_chain attribute not found')

        # Generate labels
        ndim = self.__sampler.chain.shape[-1]
        if labels is None:
            labels = np.array([f'Parameter {i + 1}' for i in range(ndim)])
        samples = self.__sampler.chain[:, burn_in:, :].reshape((-1, ndim)).copy()

        # Generate corner plot
        fig, ax = plt.subplots(ndim, ndim, figsize=figsize)
        fig = corner.corner(samples, bins=bins, labels=labels, fig=fig, **kwargs)

        return fig, ax

    def plot_posterior(self, rp, burn_in, alpha=.95, plot=True, kernel_steps=1000, bins=100):
        """
        Returns posterior distribution of return value for a specific return period.
        Can be used to explore the posterior distribution p(rv|self.extremes).

        Parameters
        ----------
        rp : float
            Return period (1/rp represents probability of exceedance over self.block_size).
        burn_in : int
            Number of samples to discard. Samples, before the series converges, should be discarded.
        alpha : float, optional
            Shows confidence bounds for given interval alpha (default=.95). Doesn't show if None.
        plot : bool, optional
            If True, plots histogram of return value (default=True). If False, return data
        kernel_steps : int, optional
            Number of bins (kernel support points) used to plot kernel density (default=1000).
        bins : int, optional
            Number of bins in historgram (default=100). Only when plot=True.

        Returns
        -------
        Distribution of return value for a given return period
            if plot = True : tuple(fig, ax)
            if plot = Fale : np.ndarray
        """

        # Make sure self.mcmc_chain exists
        if self.mcmc_chain is None:
            raise RuntimeError('No mcmc_chain attribute found.')

        if not np.isscalar(rp):
            raise ValueError('rp must be scalar')

        distribution_object = getattr(scipy.stats, self.distribution_name)

        # Calculate return value for each fit parameters sample
        ndim = self.mcmc_chain.shape[-1]
        samples = self.mcmc_chain[:, burn_in:, :].reshape((-1, ndim))
        if self.extremes_type == 'high':
            return_values = np.array(
                [
                    self.threshold + distribution_object.isf(
                        1 / rp / self.extremes_rate, *theta
                    ) for theta in samples
                ]
            )
        else:
            return_values = np.array(
                [
                    self.threshold - distribution_object.isf(
                        1 / rp / self.extremes_rate, *theta
                    ) for theta in samples
                ]
            )

        # Set up gaussian kernel
        support = np.linspace(return_values.min(), return_values.max(), kernel_steps)
        kernel = scipy.stats.gaussian_kde(return_values)
        density = kernel.evaluate(support)

        if plot:
            with plt.style.context('bmh'):
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.hist(
                    return_values, bins=bins, density=True,
                    color='k', rwidth=.9, alpha=0.2, zorder=5
                )
                ax.hist(
                    return_values, bins=bins, density=True,
                    color='k', rwidth=.9, edgecolor='k', facecolor='None', lw=.5, ls='--', zorder=10
                )
                ax.plot(
                    support, density,
                    color='k', lw=2, zorder=15
                )
                if alpha is not None:
                    ax.axvline(np.nanquantile(return_values, (1 - alpha) / 2), lw=1, color='k', ls='--')
                    ax.axvline(np.nanquantile(return_values, (1 + alpha) / 2), lw=1, color='k', ls='--')
                if self.extremes_type == 'high':
                    ax.set_xlim(right=np.nanquantile(return_values, .999))
                else:
                    ax.set_xlim(left=np.nanquantile(return_values, .001))
                ax.set_title(f'{rp}-year Return Period Posterior Distribution')
                ax.set_xlabel('Return value')
                ax.set_ylabel('Probability density')
                fig.tight_layout()
                return fig, ax
        else:
            return return_values

    def return_value(self, rp, **kwargs):
        """
        Calculates return values for given return periods.

        Parameters
        ----------
        rp : float or array_like
            Return periods (1/rp represents probability of exceedance over self.block_size).
        kwargs
            if fit is MCMC
                burn_in : int
                    Number of samples to discard. Samples, before the series converges, should be discarded.
                estimate_method : str, optional
                    'parameter mode' (default) - calculates value for parameters
                        estimated as mode (histogram peak, through gaussian kernel)
                    'value mode' - calculates values for each sample and then determines
                        value estimate as mode (histogram peak, through gaussian kernel)
                    'value quantile' - calculates values for each sample and then determines
                        value estimate as quantile of the value distribution
                kernel_steps : int, optional
                    Number of bins (kernel support points) to determine mode (default=1000).
                    Only for 'parameter mode' and 'value mode' methods.
                quantile : float, optional
                    Quantile for 'value quantile' method (default=.5, aka median).
                    Must be in the range (0, 1].

        Returns
        -------
        float or array of floats
            Return values for given return periods.
        """

        return self.isf(1 / rp / self.extremes_rate, **kwargs)

    def confidence_interval(self, rp, alpha=.95, **kwargs):
        """
        Estimates confidence intervals for given return periods.

        Parameters
        ----------
        rp : float or array_like, optional
            Return periods (1/rp represents probability of exceedance over self.block_size).
        alpha : float, optional
            Confidence interval bounds (default=.95).
        kwargs
            if fit is MCMC
                burn_in : int
                    Number of samples to discard. Samples, before the series converges, should be discarded.
            if fit is MLE
                method : str, optional
                    Confidence interval estimation method (default='Monte Carlo').
                    Supported methods:
                        'Monte Carlo' - performs many random simulations to estimate return value distribution
                        'Delta' - delta method (assumption of asymptotic normality, fast but inaccurate)
                            Implemented only for specific distributions
                        'Profile Likelihood' - not yet implemented
                if method is Monte Carlo
                    k : int, optional
                        Numeber of Monte Carlo simulations (default=1e4). Larger values result in slower simulation.
                    sampling_method : str, optional
                        Sampling method (default='constant'):
                            'constant' - number of extremes in each sample is constant and equal to len(self.extremes)
                            'poisson' - number of extremes is Poisson-distributed
                            'jacknife' - aka drop-one-out, works only when <source=data>
                    source : str, optional
                        Specifies where new data is sampled from (default='data'):
                            'data' - samples with replacement directly from extracted extreme values
                            'parametric' - samples from distribution with previously estimated (MLE) parameters
                    assume_normality : bool, optional
                        If True, assumes return values are normally distributed.
                        If False, estimates quantiles directly (default=False).
                if method is Delta
                    dx : str, optional
                        String representing a float, which represents spacing at which partial derivatives
                        are estimated (default='1e-10' for GPD and GEV, '1e-6' for others).
                    precision : int, optional
                        Precision of floating point calculations (see mpmath library documentation) (default=100).
                        Derivative estimated with low <precision> value may have
                        a significant error due to rounding and under-/overflow.

        Returns
        -------
        tuple of np.ndarray objects
            Tuple with arrays with confidence intervals (lower, upper).
        """

        # Make sure fit method was executed and fit data was generated
        if not self.__status['fit']:
            raise ValueError('No fit information found. Run self.fit() method before generating confidence intervals')

        if self.fit_method == 'MLE':
            method = kwargs.pop('method', 'Monte Carlo')

            if method == 'Monte Carlo':
                return self.__monte_carlo(rp=rp, alpha=alpha, **kwargs)

            elif method == 'Delta':
                return self.__delta(rp=rp, alpha=alpha, **kwargs)

            elif method in ['Profile Likelihood']:
                # TODO - implement Profile Likelihood mehtod
                raise NotImplementedError(f'Method {method} not implemented')

            else:
                raise ValueError(f'Method {method} not recognized')

        elif self.fit_method == 'MCMC':
            burn_in = kwargs.pop('burn_in')
            alpha = kwargs.pop('alpha', .95)
            assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

            distribution_object = getattr(scipy.stats, self.distribution_name)

            # Calculate return values for each fit parameters sample
            ndim = self.mcmc_chain.shape[-1]
            samples = self.mcmc_chain[:, burn_in:, :].reshape((-1, ndim))
            if self.extremes_type == 'high':
                return_values = np.array(
                    [
                        self.threshold + distribution_object.isf(
                            1 / rp / self.extremes_rate, *theta
                        ) for theta in samples
                    ]
                )
            else:
                return_values = np.array(
                    [
                        self.threshold - distribution_object.isf(
                            1 / rp / self.extremes_rate, *theta
                        ) for theta in samples
                    ]
                )

            # Calculate quantiles for lower and upper confidence bounds for each return period
            if np.isscalar(rp):
                return (
                    np.nanquantile(a=return_values.flatten(), q=(1 - alpha) / 2),
                    np.nanquantile(a=return_values.flatten(), q=(1 + alpha) / 2)
                )
            else:
                return np.array(
                    [
                        [np.nanquantile(a=row, q=(1 - alpha) / 2) for row in return_values.T],
                        [np.nanquantile(a=row, q=(1 + alpha) / 2) for row in return_values.T]
                    ]
                )

        else:
            raise RuntimeError(f'Unknown fit_method {self.fit_method} encountered')

    def __monte_carlo(self, rp, alpha=.95, **kwargs):
        """
        Runs the Monte Carlo confidence interval estimation method.

        Parameters
        ----------
        rp : float or array_like
            Return periods (1/rp represents probability of exceedance over self.block_size).
        alpha : float, optional
            Confidence interval bounds (default=.95).
        kwargs
            k : int, optional
                Numeber of Monte Carlo simulations (default=1e4). Larger values result in slower simulation.
            sampling_method : str, optional
                Sampling method (default='constant'):
                    'constant' - number of extremes in each sample is constant and equal to len(self.extremes)
                    'poisson' - number of extremes is Poisson-distributed
                    'jacknife' - aka drop-one-out, works only when <source=data>
            source : str, optional
                Specifies where new data is sampled from (default='data'):
                    'data' - samples with replacement directly from extracted extreme values
                    'parametric' - samples from distribution with previously estimated (MLE) parameters
            assume_normality : bool, optional
                If True, assumes return values are normally distributed.
                If False, estimates quantiles directly (default=False).

        Returns
        -------
        tuple of np.ndarray objects
            Tuple with arrays with confidence intervals (lower, upper).
        """

        k = kwargs.pop('k', 1e4)
        sampling_method = kwargs.pop('sampling_method', 'constant')
        source = kwargs.pop('source', 'data')
        assume_normality = kwargs.pop('assume_normality', False)
        # TODO - implement a discard rule (discard bad samples)
        # discard_rule = kwargs.pop('discard_rule', None)
        assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

        distribution_object = getattr(scipy.stats, self.distribution_name)
        exceedances = self.extremes[self.column].values - self.threshold
        if self.extremes_type == 'low':
            exceedances *= -1

        # Sample from data case
        if source == 'data':

            if sampling_method == 'constant':
                sample_size = len(self.extremes)
                return_values = []
                while len(return_values) < k:
                    sample = np.random.choice(a=exceedances, size=sample_size, replace=True)
                    sample_fit_parameters = distribution_object.fit(sample, **self.scipy_fit_options)
                    if self.extremes_type == 'high':
                        return_values.append(
                            self.threshold + distribution_object.isf(
                                1 / rp / self.extremes_rate, *sample_fit_parameters
                            )
                        )
                    else:
                        return_values.append(
                            self.threshold - distribution_object.isf(
                                1 / rp / self.extremes_rate, *sample_fit_parameters
                            )
                        )

            elif sampling_method == 'poisson':
                return_values = []
                while len(return_values) < k:
                    sample_size = scipy.stats.poisson.rvs(mu=len(self.extremes), loc=0, size=1)
                    sample_rate = sample_size / self.number_of_blocks
                    sample = np.random.choice(a=exceedances, size=sample_size, replace=True)
                    sample_fit_parameters = distribution_object.fit(sample, **self.scipy_fit_options)
                    if self.extremes_type == 'high':
                        return_values.append(
                            self.threshold + distribution_object.isf(
                                1 / rp / sample_rate, *sample_fit_parameters
                            )
                        )
                    else:
                        return_values.append(
                            self.threshold - distribution_object.isf(
                                1 / rp / sample_rate, *sample_fit_parameters
                            )
                        )

            elif sampling_method == 'jacknife':
                sample_rate = (len(self.extremes) - 1) / self.number_of_blocks
                return_values = []
                for i in range(len(self.extremes)):
                    sample = np.delete(arr=exceedances, obj=i)
                    sample_fit_parameters = distribution_object.fit(sample, **self.scipy_fit_options)
                    if self.extremes_type == 'high':
                        return_values.append(
                            self.threshold + distribution_object.isf(
                                1 / rp / sample_rate, *sample_fit_parameters
                            )
                        )
                    else:
                        return_values.append(
                            self.threshold - distribution_object.isf(
                                1 / rp / sample_rate, *sample_fit_parameters
                            )
                        )

            else:
                raise ValueError(f'for <source=data> the sampling method must be <constant>, <poisson>, or <jacknife>,'
                                 f' <{sampling_method}> was passed')

        # Sample from distribution (parametric) case
        elif source == 'parametric':

            if sampling_method == 'constant':
                sample_size = len(self.extremes)
                return_values = []
                while len(return_values) < k:
                    sample = distribution_object.rvs(*self.fit_parameters, size=sample_size)
                    sample_fit_parameters = distribution_object.fit(sample, **self.scipy_fit_options)
                    if self.extremes_type == 'high':
                        return_values.append(
                            self.threshold + distribution_object.isf(
                                1 / rp / self.extremes_rate, *sample_fit_parameters
                            )
                        )
                    else:
                        return_values.append(
                            self.threshold - distribution_object.isf(
                                1 / rp / self.extremes_rate, *sample_fit_parameters
                            )
                        )

            elif sampling_method == 'poisson':
                return_values = []
                while len(return_values) < k:
                    sample_size = scipy.stats.poisson.rvs(mu=len(self.extremes), loc=0, size=1)
                    sample_rate = sample_size / self.number_of_blocks
                    sample = distribution_object.rvs(*self.fit_parameters, size=sample_size)
                    sample_fit_parameters = distribution_object.fit(sample, **self.scipy_fit_options)
                    if self.extremes_type == 'high':
                        return_values.append(
                            self.threshold + distribution_object.isf(
                                1 / rp / sample_rate, *sample_fit_parameters
                            )
                        )
                    else:
                        return_values.append(
                            self.threshold - distribution_object.isf(
                                1 / rp / sample_rate, *sample_fit_parameters
                            )
                        )

            else:
                raise ValueError(f'for <source=parametric> the sampling method must be <constant> or <poisson>,'
                                 f' <{sampling_method}> was passed')

        else:
            raise ValueError(f'source must be either <data> or <parametric>, <{source}> was passed')

        # Estimate confidence bounds for sampled return values
        return_values = np.array(return_values)
        if np.isscalar(rp):
            if assume_normality:
                return scipy.stats.norm.interval(
                    alpha=alpha, loc=np.nanmean(return_values), scale=np.nanstd(return_values, ddof=1)
                )
            else:
                return (
                    np.nanquantile(a=return_values.flatten(), q=(1 - alpha) / 2),
                    np.nanquantile(a=return_values.flatten(), q=(1 + alpha) / 2)
                )
        else:
            if assume_normality:
                locations = np.array([np.nanmean(row) for row in return_values.T])
                scales = np.array([np.nanstd(row, ddof=1) for row in return_values.T])
                return np.transpose(
                    [
                        scipy.stats.norm.interval(alpha=alpha, loc=loc, scale=scale)
                        for loc, scale in zip(locations, scales)
                    ]
                )
            else:
                return np.array(
                    [
                        [np.nanquantile(a=row, q=(1 - alpha) / 2) for row in return_values.T],
                        [np.nanquantile(a=row, q=(1 + alpha) / 2) for row in return_values.T]
                    ]
                )

    def __delta(self, rp, alpha=.95, **kwargs):
        """
        Estimates confidence intervals using the delta method. Assumes asymptotic normality.

        Parameters
        ----------
        rp : float or array_like
            Return periods (1/rp represents probability of exceedance over self.block_size).
        alpha : float, optional
            Confidence interval bounds (default=.95).
        kwargs
            dx : str, optional
                String representing a float, which represents spacing at which partial derivatives
                are estimated (default='1e-10').
            precision : int, optional
                Precision of floating point calculations (see mpmath library documentation) (default=100).
                Derivative estimated with low <precision> value may have
                a significant error due to rounding and under-/overflow.

        Returns
        -------
        tuple of np.ndarray objects
            Tuple with arrays with confidence intervals (lower, upper).
        """

        dx = kwargs.pop('dx', '1e-10')
        precision = kwargs.pop('precision', 100)
        assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

        # Make sure fit method was executed and fit data was generated
        if not self.__status['fit']:
            raise ValueError('No fit information found. Run self.fit() method before generating confidence intervals')

        # Check if a custom distribution with mpmath backend is defined
        if self.distribution_name in coastlib.stats.distributions.distributions:
            distribution_object = getattr(coastlib.stats.distributions, self.distribution_name)
        else:
            raise ValueError(f'Delta method is not implemented for {self.distribution_name} distribution')

        # Account for custom fit parameters (custom genextreme has negative shape in scipy)
        if self.distribution_name == 'genextreme':
            fit_parameters = self.fit_parameters * np.array([-1, 1, 1])
        elif self.distribution_name in ['genpareto']:
            fit_parameters = self.fit_parameters
        else:
            raise ValueError(f'Delta method is not implemented for {self.distribution_name} distribution')

        exceedances = self.extremes[self.column].values - self.threshold
        # Flip exceedances around 0
        if self.extremes_type == 'low':
            exceedances *= -1

        # Generalized Pareto Distribution
        if self.distribution_name == 'genpareto':
            if self.scipy_fit_options != dict(floc=0):
                raise ValueError(
                    f'Delta method for genpareto is implemented only for the case of '
                    f'fixed location parameter {dict(floc=0)}, '
                    f'{self.scipy_fit_options} does not satisfy this criteria'
                )

            with mpmath.workdps(precision):
                # Define modified log_likehood function (only shape and scale, location is fixed)
                def log_likelihood(*theta):
                    return mpmath.fsum(
                        [
                            mpmath.log(
                                coastlib.stats.distributions.genpareto.pdf(
                                    x=x, shape=theta[0], loc=fit_parameters[1], scale=theta[1]
                                )
                            ) for x in exceedances
                        ]
                    )

                # Calculate covariance matrix of shape and scale
                observed_information = -coastlib.math.derivatives.hessian(
                    func=log_likelihood, n=2, dx=dx, precision=precision,
                    coordinates=(fit_parameters[0], fit_parameters[2])
                ).astype(np.float64)
                covariance = np.linalg.inv(observed_information)

                # Modify covariance matrix to include uncertainty in threshold exceedance probability
                modified_covariance = np.zeros((3, 3))
                modified_covariance[1:, 1:] = covariance
                # Probability of exceeding threshold for all observations
                eta_0 = len(self.extremes) / len(self.dataframe)
                # Number of observations per year
                ny = len(self.dataframe) / self.number_of_blocks
                modified_covariance[0][0] = eta_0 * (1 - eta_0) / len(self.dataframe)

            if np.isscalar(rp):
                # Define scalar function as a function which takes arbitrary fit parameters and returns return values
                def scalar_function(eta, *theta):
                    q = 1 / (rp * ny * eta)
                    if q <= 0 or q >= 1:
                        return np.nan
                    if self.extremes_type == 'high':
                        return self.threshold + distribution_object.isf(
                            q=q, shape=theta[0], loc=fit_parameters[1], scale=theta[1]
                        )
                    else:
                        return self.threshold - distribution_object.isf(
                            q=q, shape=theta[0], loc=fit_parameters[1], scale=theta[1]
                        )

                delta_scalar = coastlib.math.derivatives.gradient(
                    func=scalar_function, n=3, dx=dx, precision=precision,
                    coordinates=(eta_0, fit_parameters[0], fit_parameters[2])
                )

                loc = np.float64(
                    scalar_function(eta_0, fit_parameters[0], fit_parameters[2])
                )
                variance = np.dot(
                    np.dot(delta_scalar.T, modified_covariance), delta_scalar
                ).flatten().astype(np.float64)[0]

                return scipy.stats.norm.interval(alpha=alpha, loc=loc, scale=np.sqrt(variance))

            else:
                locs, variances = [], []
                for _rp in rp:
                    # Define scalar function as a function which takes arbitrary fit parameters
                    # and returns return values
                    def scalar_function(eta, *theta):
                        q = 1 / (_rp * ny * eta)
                        if q <= 0 or q >= 1:
                            return np.nan
                        if self.extremes_type == 'high':
                            return self.threshold + distribution_object.isf(
                                q=q, shape=theta[0], loc=fit_parameters[1], scale=theta[1]
                            )
                        else:
                            return self.threshold - distribution_object.isf(
                                q=q, shape=theta[0], loc=fit_parameters[1], scale=theta[1]
                            )

                    delta_scalar = coastlib.math.derivatives.gradient(
                        func=scalar_function, n=3, dx=dx, precision=precision,
                        coordinates=(eta_0, fit_parameters[0], fit_parameters[2]),
                    )

                    locs.append(
                        np.float64(
                            scalar_function(eta_0, fit_parameters[0], fit_parameters[2])
                        )
                    )
                    variances.append(
                        np.dot(
                            np.dot(delta_scalar.T, modified_covariance), delta_scalar
                        ).flatten().astype(np.float64)[0]
                    )
                return np.array(
                    [
                        scipy.stats.norm.interval(alpha=alpha, loc=loc, scale=np.sqrt(variance))
                        for loc, variance in zip(locs, variances)
                    ]
                ).T

        # Generalized Extreme Distribtuion
        elif self.distribution_name == 'genextreme':
            if self.scipy_fit_options != {}:
                raise ValueError(
                    f'Delta method for genextreme is implemented only for the case of '
                    f'unbound parameters {dict()}, '
                    f'{self.scipy_fit_options} does not satisfy this criteria'
                )

            # Calculate observed information matrix (negative hessian of log_likelihood)
            observed_information = distribution_object.observed_information(
                exceedances, *fit_parameters, dx=dx, precision=precision
            ).astype(np.float64)

            if np.isscalar(rp):
                # Define scalar function as a function which takes arbitrary fit parameters and returns return values
                def scalar_function(*theta):
                    q = 1 / rp / self.extremes_rate
                    if q <= 0 or q >= 1:
                        return np.nan
                    if self.extremes_type == 'high':
                        return self.threshold + distribution_object.isf(q, *theta)
                    else:
                        return self.threshold - distribution_object.isf(q, *theta)

                # Calculate delta (gradient) of scalar_function
                delta_scalar = coastlib.math.derivatives.gradient(
                    func=scalar_function, n=len(fit_parameters),
                    coordinates=fit_parameters, dx=dx, precision=precision
                ).astype(np.float64)

                # Calculate location and scale (gaussian mean and sigma)
                loc = np.float64(scalar_function(*fit_parameters))
                variance = np.dot(
                    np.dot(delta_scalar.T, np.linalg.inv(observed_information)), delta_scalar
                ).flatten()[0]

                return scipy.stats.norm.interval(alpha=alpha, loc=loc, scale=np.sqrt(variance))

            else:
                locs, variances = [], []
                for _rp in rp:
                    # Define scalar function as a function which takes arbitrary fit parameters
                    # and returns return values
                    def scalar_function(*theta):
                        q = 1 / _rp / self.extremes_rate
                        if q <= 0 or q >= 1:
                            return np.nan
                        if self.extremes_type == 'high':
                            return self.threshold + distribution_object.isf(q, *theta)
                        else:
                            return self.threshold - distribution_object.isf(q, *theta)

                    # Calculate delta (gradient) of scalar_function
                    delta_scalar = coastlib.math.derivatives.gradient(
                        func=scalar_function, n=len(fit_parameters),
                        coordinates=fit_parameters, dx=dx, precision=precision
                    ).astype(np.float64)

                    # Calculate location and scale (gaussian mean and sigma)
                    locs.append(np.float64(scalar_function(*fit_parameters)))
                    variances.append(
                        np.dot(
                            np.dot(delta_scalar.T, np.linalg.inv(observed_information)), delta_scalar
                        ).flatten()[0]
                    )
                return np.array(
                    [
                        scipy.stats.norm.interval(alpha=alpha, loc=loc, scale=np.sqrt(variance))
                        for loc, variance in zip(locs, variances)
                    ]
                ).T

    def generate_results(self, rp=None, alpha=.95, **kwargs):
        """
        Generates a self.results dataframe with return values and, optionally, confidence intervals.
        Used to generate data for output and reporting purpose (run the self.restuls.to_excel()) and to
        produce a probability plot (summary).

        Parameters
        ----------
        rp : float or array_like, optional
            Return periods (1/rp represents probability of exceedance over self.block_size).
            By default is an array of return periods equally spaced on a log-scale from 0.001 to 1000.
        alpha : float, optional
            Confidence interval bounds (default=.95). Doesn't estimate confidence intervals if None.
        kwargs
            if fit is MCMC:
                rv_kwargs : dict
                    burn_in : int
                        Number of samples to discard. Samples, before the series converges, should be discarded.
                    estimate_method : str, optional
                        'parameter mode' (default) - calculates value for parameters
                            estimated as mode (histogram peak, through gaussian kernel)
                        'value mode' - calculates values for each sample and then determines
                            value estimate as mode (histogram peak, through gaussian kernel)
                        'value quantile' - calculates values for each sample and then determines
                            value estimate as quantile of the value distribution
                    kernel_steps : int, optional
                        Number of bins (kernel support points) to determine mode (default=1000).
                        Only for 'parameter mode' and 'value mode' methods.
                    quantile : float, optional
                        Quantile for 'value quantile' method (default=.5, aka median).
                        Must be in the range (0, 1].
                ci_kwargs : dict
                    burn_in : int
                        Number of samples to discard. Samples, before the series converges, should be discarded.
            if fit is MLE
                ci_kwargs
                    method : str, optional
                        Confidence interval estimation method (default='Monte Carlo').
                        Supported methods:
                            'Monte Carlo' - performs many random simulations to estimate return value distribution
                            'Delta' - delta method (assumption of asymptotic normality, fast but inaccurate)
                                Implemented only for specific distributions
                            'Profile Likelihood' - not yet implemented
                    if method is Monte Carlo
                        k : int, optional
                            Numeber of Monte Carlo simulations (default=1e4). Larger values result in slower simulation.
                        sampling_method : str, optional
                            Sampling method (default='constant'):
                                'constant' - number of extremes in each sample is constant and equal to
                                    number of extracted extreme values
                                'poisson' - number of extremes is Poisson-distributed
                                'jacknife' - aka drop-one-out, works only when <source=data>
                        source : str, optional
                            Specifies where new data is sampled from (default='data'):
                                'data' - samples with replacement directly from extracted extreme values
                                'parametric' - samples from distribution with previously estimated (MLE) parameters
                        assume_normality : bool, optional
                            If True, assumes return values are normally distributed.
                            If False, estimates quantiles directly (default=False).
                    if method is Delta
                        dx : str, optional
                            String representing a float, which represents spacing at which partial derivatives
                            are estimated (default='1e-10' for GPD and GEV, '1e-6' for others).
                        precision : int, optional
                            Precision of floating point calculations (see mpmath library documentation) (default=100).
                            Derivative estimated with low <precision> value may have
                            a significant error due to rounding and under-/overflow.

        Returns
        -------
        Creates a <self.results> dataframe with return values and, optionally, confidence intervals
        for each given return period.
        """

        # Make sure fit method was executed and fit data was generated
        if not self.__status['fit']:
            raise ValueError('No fit information found. Run self.fit() method first')

        if rp is None:
            rp = np.unique(
                np.append(
                    np.logspace(-3, 3, 200),
                    [1/12, 7/365.2425, 1, 2, 5, 10, 25, 50, 100, 200, 250, 500, 1000]
                )
            )

        # Update internal status
        self.__status = dict(
            extremes=True,
            fit=True,
            results=False
        )
        self.__update()

        if np.isscalar(rp):
            rp = np.array([rp])
        else:
            rp = np.array(rp)

        if self.fit_method == 'MLE':
            rv_kwargs = kwargs.pop('rv_kwargs', {})
            ci_kwargs = kwargs.pop('ci_kwargs', {})
        else:
            rv_kwargs = kwargs.pop('rv_kwargs')
            ci_kwargs = kwargs.pop('ci_kwargs')
        assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

        return_values = self.return_value(rp, **rv_kwargs)

        self.results = pd.DataFrame(
            data=return_values, index=rp, columns=['Return Value']
        )
        self.results.index.name = 'Return Period'

        if alpha is not None:
            ci_lower, ci_upper = self.confidence_interval(rp=rp, alpha=alpha, **ci_kwargs)
            if np.isscalar(ci_lower):
                ci_lower, ci_upper = np.array([ci_lower]), np.array([ci_upper])
            else:
                ci_lower, ci_upper = np.array(ci_lower), np.array(ci_upper)
            self.results[f'{alpha*100:.0f}% CI Lower'] = ci_lower
            self.results[f'{alpha*100:.0f}% CI Upper'] = ci_upper

        # Remove bad values from the results
        if self.extremes_type == 'high':
            mask = self.results['Return Value'].values >= self.extremes[self.column].values.min()
        else:
            mask = self.results['Return Value'].values <= self.extremes[self.column].values.max()
        self.results = self.results[mask]

        # Update internal status
        self.__status = dict(
            extremes=True,
            fit=True,
            results=True
        )
        self.__update()

    def pdf(self, x, **kwargs):
        """
        Estimates probability density at value <x> using the fitted distribution.

        Parameters
        ----------
        x : float or iterable
            Values at which the probability density is estimated.
        kwargs
            if fit is MCMC
                burn_in : int
                    Number of samples to discard. Samples, before the series converges, should be discarded.
                estimate_method : str, optional
                    'parameter mode' (default) - calculates value for parameters
                        estimated as mode (histogram peak, through gaussian kernel)
                    'value mode' - calculates values for each sample and then determines
                        value estimate as mode (histogram peak, through gaussian kernel)
                    'value quantile' - calculates values for each sample and then determines
                        value estimate as quantile of the value distribution
                kernel_steps : int, optional
                    Number of bins (kernel support points) to determine mode (default=1000).
                    Only for 'parameter mode' and 'value mode' methods.
                quantile : float, optional
                    Quantile for 'value quantile' method (default=.5, aka median).
                    Must be in the range (0, 1].

        Returns
        -------
        Depending on x, either estimate or array of estimates of probability densities at <x>.
        """

        if self.extremes_type == 'high':
            return self.___get_property(x=x-self.threshold, prop='pdf', **kwargs)
        else:
            return self.___get_property(x=self.threshold-x, prop='pdf', **kwargs)

    def cdf(self, x, **kwargs):
        """
        Estimates cumulative probability at value <x> using the fitted distribution.

        Parameters
        ----------
        x : float or iterable
            Values at which the cumulative probability density is estimated.
        kwargs
            if fit is MCMC
                burn_in : int
                    Number of samples to discard. Samples, before the series converges, should be discarded.
                estimate_method : str, optional
                    'parameter mode' (default) - calculates value for parameters
                        estimated as mode (histogram peak, through gaussian kernel)
                    'value mode' - calculates values for each sample and then determines
                        value estimate as mode (histogram peak, through gaussian kernel)
                    'value quantile' - calculates values for each sample and then determines
                        value estimate as quantile of the value distribution
                kernel_steps : int, optional
                    Number of bins (kernel support points) to determine mode (default=1000).
                    Only for 'parameter mode' and 'value mode' methods.
                quantile : float, optional
                    Quantile for 'value quantile' method (default=.5, aka median).
                    Must be in the range (0, 1].

        Returns
        -------
        Depending on x, either estimate or array of estimates of cumulative probability at <x>.
        """

        if self.extremes_type == 'high':
            return self.___get_property(x=x-self.threshold, prop='cdf', **kwargs)
        else:
            return self.___get_property(x=self.threshold-x, prop='cdf', **kwargs)

    def ppf(self, q, **kwargs):
        """
        Estimates ppf (inverse cdf or quantile function) at value <x> using the fitted distribution.

        Parameters
        ----------
        q : float or iterable
            Quantiles at which the ppf is estimated.
        kwargs
            if fit is MCMC
                burn_in : int
                    Number of samples to discard. Samples, before the series converges, should be discarded.
                estimate_method : str, optional
                    'parameter mode' (default) - calculates value for parameters
                        estimated as mode (histogram peak, through gaussian kernel)
                    'value mode' - calculates values for each sample and then determines
                        value estimate as mode (histogram peak, through gaussian kernel)
                    'value quantile' - calculates values for each sample and then determines
                        value estimate as quantile of the value distribution
                kernel_steps : int, optional
                    Number of bins (kernel support points) to determine mode (default=1000).
                    Only for 'parameter mode' and 'value mode' methods.
                quantile : float, optional
                    Quantile for 'value quantile' method (default=.5, aka median).
                    Must be in the range (0, 1].

        Returns
        -------
        Depending on x, either estimate or array of estimates of ppf at <x>.
        """

        if self.extremes_type == 'high':
            return self.threshold + self.___get_property(x=q, prop='ppf', **kwargs)
        else:
            return self.threshold - self.___get_property(x=q, prop='ppf', **kwargs)

    def isf(self, q, **kwargs):
        """
        Estimates isf (inverse survival or upper quantile function) at value <x> using the fitted distribution.

        Parameters
        ----------
        q : float or iterable
            Quantiles at which the isf is estimated.
        kwargs
            if fit is MCMC
                burn_in : int
                    Number of samples to discard. Samples, before the series converges, should be discarded.
                estimate_method : str, optional
                    'parameter mode' (default) - calculates value for parameters
                        estimated as mode (histogram peak, through gaussian kernel)
                    'value mode' - calculates values for each sample and then determines
                        value estimate as mode (histogram peak, through gaussian kernel)
                    'value quantile' - calculates values for each sample and then determines
                        value estimate as quantile of the value distribution
                kernel_steps : int, optional
                    Number of bins (kernel support points) to determine mode (default=1000).
                    Only for 'parameter mode' and 'value mode' methods.
                quantile : float, optional
                    Quantile for 'value quantile' method (default=.5, aka median).
                    Must be in the range (0, 1].

        Returns
        -------
        Depending on x, either estimate or array of estimates of isf at <x>.
        """

        if self.extremes_type == 'high':
            return self.threshold + self.___get_property(x=q, prop='isf', **kwargs)
        else:
            return self.threshold - self.___get_property(x=q, prop='isf', **kwargs)

    def ___get_property(self, x, prop, **kwargs):
        """
        Estimates property (pdf, cdf, ppf, etc.) at value <x> using the fitted distribution parameters.

        Parameters
        ----------
        x : float or iterable
            Value at which the property is estimated.
        prop : str
            Scipy property to be estimated (pdf, ppf, isf, cdf, rvs, etc.).
        kwargs
            if fit is MCMC
                burn_in : int
                    Number of samples to discard. Samples, before the series converges, should be discarded.
                estimate_method : str, optional
                    'parameter mode' (default) - calculates value for parameters
                        estimated as mode (histogram peak, through gaussian kernel)
                    'value mode' - calculates values for each sample and then determines
                        value estimate as mode (histogram peak, through gaussian kernel)
                    'value quantile' - calculates values for each sample and then determines
                        value estimate as quantile of the value distribution
                kernel_steps : int, optional
                    Number of bins (kernel support points) to determine mode (default=1000).
                    Only for 'parameter mode' and 'value mode' methods.
                quantile : float, optional
                    Quantile for 'value quantile' method (default=.5, aka median).
                    Must be in the range (0, 1].

        Returns
        -------
        Depending on x, either estimate or array of estimates of property at <x>
        """

        # Make sure fit method was executed and fit data was generated
        if not self.__status['fit']:
            raise ValueError('No fit information found. Run self.fit() method first')

        distribution_object = getattr(scipy.stats, self.distribution_name)
        property_function = getattr(distribution_object, prop)
        if not np.isscalar(x):
            x = np.array(x)

        if self.fit_method == 'MLE':
            assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'
            return property_function(x, *self.fit_parameters)

        elif self.fit_method == 'MCMC':
            burn_in = kwargs.pop('burn_in')
            estimate_method = kwargs.pop('estimate_method', 'parameter mode')
            if estimate_method not in ['parameter mode', 'value mode', 'value quantile']:
                raise ValueError(f'Estimate method <{estimate_method}> not recognized')
            if estimate_method in ['parameter mode', 'value mode']:
                kernel_steps = kwargs.pop('kernel_steps', 1000)
            else:
                kernel_steps = None
            if estimate_method == 'value quantile':
                quantile = kwargs.pop('quantile', .5)
            else:
                quantile = None
            assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

            # Estimate mode of each parameter as peaks of gaussian kernel.
            # Use estimated parameters to calculate property function
            if estimate_method == 'parameter mode':
                parameters = self._kernel_fit_parameters(burn_in=burn_in, kernel_steps=kernel_steps)
                return property_function(x, *parameters)

            # Load samples
            ndim = self.mcmc_chain.shape[-1]
            samples = self.mcmc_chain[:, burn_in:, :].reshape((-1, ndim))

            property_samples = np.array([property_function(x, *_theta) for _theta in samples])

            # Estimate property function as mode of distribution of property value
            # for all samples in self.mcmc_chain as peaks of gaussian kernel.
            if estimate_method == 'value mode':
                if np.isscalar(x):
                    if np.all(np.isnan(property_samples)):
                        return np.nan
                    else:
                        ps_filtered = property_samples[~np.isnan(property_samples)]
                        if np.all(ps_filtered == ps_filtered[0]):
                            return np.nan
                        else:
                            kernel = scipy.stats.gaussian_kde(ps_filtered)
                            support = np.linspace(ps_filtered.min(), ps_filtered.max(), kernel_steps)
                            density = kernel.evaluate(support)
                            return support[density.argmax()]
                else:
                    estimates = []
                    for ps in property_samples.T:
                        if np.all(np.isnan(ps)):
                            estimates.append(np.nan)
                        else:
                            ps_filtered = ps[~np.isnan(ps)]
                            if np.all(ps_filtered == ps_filtered[0]):
                                estimates.append(np.nan)
                            else:
                                kernel = scipy.stats.gaussian_kde(ps_filtered)
                                support = np.linspace(ps_filtered.min(), ps_filtered.max(), kernel_steps)
                                density = kernel.evaluate(support)
                                estimates.append(support[density.argmax()])
                    return np.array(estimates)

            # Estimate property function as quantile of distribution of property value
            # for all samples in self.mcmc_chain.
            elif estimate_method == 'value quantile':
                if np.isscalar(quantile):
                    if quantile <= 0 or quantile > 1:
                        raise ValueError(f'Quantile must be in range (0,1], quantile={quantile} was passed')
                else:
                    raise ValueError(f'Quantile must be scalar, {type(quantile)} was passed')

                if np.isscalar(x):
                    return np.nanquantile(a=property_samples, q=quantile)
                else:
                    return np.array(
                        [
                            np.nanquantile(a=row, q=quantile) for row in property_samples.T
                        ]
                    )

        else:
            raise RuntimeError(f'Unknown fit_method {self.fit_method} encountered')

    def plot_summary(self, support=None, bins=10, plotting_position='Weibull', **kwargs):
        """
        Plots projected return values, pdf, and cdf values against observed.

        Parameters
        ----------
        support : array_like, optional
            Values used to estimate pdf and cdf. By default is 100 linearly spaced min to max extreme values.
        bins : int, optional
            Number of bins used to plot cdf and pdf histograms (default=10).
        plotting_position : str, optional
            Plotting position (default='Weibull'). Has no effect on return value inference,
            affects only some goodness of fit statistics and locations of observed extremes on the
            return values plot.
        kwargs
            if fit is MCMC:
                rv_kwargs : dict
                    burn_in : int
                        Number of samples to discard. Samples, before the series converges, should be discarded.
                    estimate_method : str, optional
                        'parameter mode' (default) - calculates value for parameters
                            estimated as mode (histogram peak, through gaussian kernel)
                        'value mode' - calculates values for each sample and then determines
                            value estimate as mode (histogram peak, through gaussian kernel)
                        'value quantile' - calculates values for each sample and then determines
                            value estimate as quantile of the value distribution
                    kernel_steps : int, optional
                        Number of bins (kernel support points) to determine mode (default=1000).
                        Only for 'parameter mode' and 'value mode' methods.
                    quantile : float, optional
                        Quantile for 'value quantile' method (default=.5, aka median).
                        Must be in the range (0, 1].

        Returns
        -------
        tuple(fig, ax1, ax2, ax3)
            Figure, return value, pdf, cdf axes.
        """

        # Make sure fit method was executed and fit data was generated
        if not self.__status['results']:
            raise UnboundLocalError('No data found. Generate results by runing self.generate_results() method first')

        if support is None:
            support = np.linspace(
                self.extremes[self.column].values.min(), self.extremes[self.column].values.max(), 100
            )

        if self.fit_method == 'MCMC':
            rv_kwargs = kwargs.pop('rv_kwargs')
        else:
            rv_kwargs = {}
        assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

        return_period = self.__get_return_period(plotting_position=plotting_position)

        with plt.style.context('bmh'):
            # Setup canvas
            fig = plt.figure(figsize=(12, 8))
            ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
            ax2 = plt.subplot2grid((2, 2), (1, 0))
            ax3 = plt.subplot2grid((2, 2), (1, 1))

            # Plot return values
            ax1.set_title('Return Value Plot')
            ax1.set_ylabel(f'{self.column}')
            ax1.set_xlabel(f'Return period')
            ax1.plot(
                self.results.index, self.results['Return Value'].values,
                color='k', lw=2, zorder=15, label='Central estimate'
            )
            if len(self.results.columns) == 3:
                ax1.plot(
                    self.results.index, self.results[self.results.columns[1]].values,
                    ls='--', color='k', lw=.5, zorder=10
                )
                ax1.plot(
                    self.results.index, self.results[self.results.columns[2]].values,
                    ls='--', color='k', lw=.5, zorder=10
                )
                ax1.fill_between(
                    self.results.index, self.results[self.results.columns[1]],
                    self.results[self.results.columns[2]],
                    alpha=.1, color='k',
                    label=f'{self.results.columns[1].split("%")[0]}% confidence interval', zorder=5
                )
            points = ax1.scatter(
                return_period, self.extremes[self.column].values,
                edgecolors='white', marker='o', facecolors='k', s=40, lw=1, zorder=15,
                label=f'Observed extreme event\n{plotting_position} plotting position'
            )
            ax1.semilogx()
            ax1.grid(b=True, which='minor', axis='x')
            ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
            ax1.legend()

            annot = ax1.annotate(
                "", xy=(self.extremes['Return Period'].values.mean(), self.extremes[self.column].values.mean()),
                xytext=(10, 10), textcoords="offset points",
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='k', lw=1, zorder=25),
                zorder=30
            )
            point = ax1.scatter(
                self.extremes['Return Period'].values.mean(), self.extremes[self.column].values.mean(),
                edgecolors='white', marker='o', facecolors='orangered', s=80, lw=1, zorder=20
            )
            point.set_visible(False)
            annot.set_visible(False)

            def update_annot(ind):
                n = ind['ind'][0]
                pos = points.get_offsets()[n]
                annot.xy = pos
                point.set_offsets(pos)
                text = str(
                    f'Date : {self.extremes.index[n]}\n'
                    f'Value : {self.extremes[self.column].values[n]:.2f}\n'
                    f'Return Period : {return_period[n]:.2f}'
                )
                annot.set_text(text)

            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax1:
                    cont, ind = points.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        point.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            point.set_visible(False)
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect('motion_notify_event', hover)

            # Plot PDF
            ax2.set_ylabel('Probability density')
            ax2.set_xlabel(f'{self.column}')
            ax2.hist(
                self.extremes[self.column].values, bins=bins, density=True,
                color='k', rwidth=.9, alpha=0.2, zorder=5
            )
            ax2.hist(
                self.extremes[self.column].values, bins=bins, density=True,
                color='k', rwidth=.9, edgecolor='k', facecolor='None', lw=1, ls='--', zorder=10
            )
            ax2.plot(
                support, self.pdf(support, **rv_kwargs),
                color='k', lw=2, zorder=15
            )
            ax2.scatter(
                self.extremes[self.column].values, [0] * len(self.extremes),
                edgecolors='white', marker='o', facecolors='k', s=40, lw=1, zorder=20
            )
            ax2.set_ylim(0)

            # Plot CDF
            ax3.set_ylabel('Cumulative probability')
            ax3.set_xlabel(f'{self.column}')
            if self.extremes_type == 'high':
                ax3.hist(
                    self.extremes[self.column], bins=bins, density=True, cumulative=True,
                    color='k', rwidth=.9, alpha=0.2, zorder=5
                )
                ax3.hist(
                    self.extremes[self.column], bins=bins, density=True, cumulative=True,
                    color='k', rwidth=.9, edgecolor='k', facecolor='None', lw=1, ls='--', zorder=10
                )
            else:
                _, boundaries = np.histogram(self.extremes[self.column].values, bins)
                centers = np.array([(boundaries[i] + boundaries[i - 1]) / 2 for i in range(1, len(boundaries))])
                densities = []
                for i, c in enumerate(centers):
                    mask = self.extremes[self.column].values >= boundaries[i]
                    densities.append(np.sum(mask) / len(self.extremes))
                ax3.bar(
                    centers, densities, width=.9*(boundaries[1]-boundaries[0]),
                    color='k', alpha=0.2, zorder=5
                )
                ax3.bar(
                    centers, densities, width=.9*(boundaries[1]-boundaries[0]),
                    color='k', edgecolor='k', facecolor='None', lw=1, ls='--', zorder=10
                )
            ax3.plot(
                support, self.cdf(support, **rv_kwargs),
                color='k', lw=2, zorder=15
            )
            ax3.scatter(
                self.extremes[self.column].values, [0] * len(self.extremes),
                edgecolors='white', marker='o', facecolors='k', s=40, lw=1, zorder=20
            )
            ax3.set_ylim(0)

            fig.tight_layout()

            return fig, ax1, ax2, ax3

    def plot_qq(self, k, plot=True, plotting_position='Weibull', quantiles=True, **kwargs):
        """
        Plots theoretical quantiles (probabilites) agains observed quantiles (probabilites).

        Parameters
        ----------
        k : int
            Number of estimated (non-fixed) parameters in the distribution.
        plot : bool, optional
            Generates plot if True, returns data if False (default=True).
        plotting_position : str, optional
            Plotting position (default='Weibull'). Has no effect on return value inference,
            affects only some goodness of fit statistics and locations of observed extremes on the
            return values plot.
        quantiles : bool, optional
            If True, produces a quantile plot (Q-Q, ppf) (default=True).
            If False, produces a probability plot (P-P, cdf).
        kwargs
            if fit is MCMC:
                rv_kwargs : dict
                    burn_in : int
                        Number of samples to discard. Samples, before the series converges, should be discarded.
                    estimate_method : str, optional
                        'parameter mode' (default) - calculates value for parameters
                            estimated as mode (histogram peak, through gaussian kernel)
                        'value mode' - calculates values for each sample and then determines
                            value estimate as mode (histogram peak, through gaussian kernel)
                        'value quantile' - calculates values for each sample and then determines
                            value estimate as quantile of the value distribution
                    kernel_steps : int, optional
                        Number of bins (kernel support points) to determine mode (default=1000).
                        Only for 'parameter mode' and 'value mode' methods.
                    quantile : float, optional
                        Quantile for 'value quantile' method (default=.5, aka median).
                        Must be in the range (0, 1].

        Returns
        -------
        if plot=True (default) : tuple(fig, ax)
        if plot=False :
            tuple((theoretical, observed), (r, p))
        """

        # Make sure fit method was executed and fit data was generated
        if not self.__status['fit']:
            raise ValueError('No fit information found. Run self.fit() method first')

        if self.fit_method == 'MLE':
            rv_kwargs = kwargs.pop('rv_kwargs', {})
        else:
            rv_kwargs = kwargs.pop('rv_kwargs')
        assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

        ecdf = self.__get_return_period(plotting_position=plotting_position, return_cdf=True)
        return_periods = self.__get_return_period(plotting_position=plotting_position)

        # Estimate theoretical values based on returned quantiles
        if quantiles:
            theoretical = self.ppf(ecdf, **rv_kwargs)
        else:
            theoretical = self.cdf(self.extremes[self.column].values, **rv_kwargs)
        theoretical[np.isinf(theoretical)] = np.nan
        mask = ~np.isnan(theoretical)
        if quantiles:
            r, p = scipy.stats.pearsonr(self.extremes[self.column].values[mask], theoretical[mask])
        else:
            r, p = scipy.stats.pearsonr(ecdf, theoretical[mask])
        r = np.sqrt(
            1 - (1 - r ** 2) * (len(theoretical[mask]) - 1) / (len(theoretical[mask]) - (k + 1))
        )

        if plot:
            with plt.style.context('bmh'):
                # Quantile plot
                if quantiles:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    points = ax.scatter(
                        theoretical, self.extremes[self.column].values,
                        edgecolors='white', marker='o', facecolors='k', s=40, lw=1, zorder=10
                    )
                    lims = ax.get_xlim(), ax.get_ylim()
                    dlims = (-1e9, 1e9)
                    ax.plot(dlims, dlims, ls='--', lw=1, zorder=5, color='k')
                    ax.set_xlim(np.min(lims), np.max(lims))
                    ax.set_ylim(np.min(lims), np.max(lims))
                    ax.set_title(r'Quantile Plot')
                    plt.xlabel(r'Theoretical quantiles')
                    plt.ylabel(rf'Observed quantiles, {plotting_position} plotting position')
                    ax.text(
                        .05, .9, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
                        s=f'$\\bar{{R}}^2$={r**2:>.2f}\np={p:>.3f}', fontsize=14,
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='k', lw=1, zorder=25)
                    )

                    annot = ax.annotate(
                        '', xy=(theoretical[0], self.extremes[self.column].values[0]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='k', lw=1, zorder=25),
                        zorder=30
                    )
                    point = ax.scatter(
                        theoretical[0]+self.threshold, self.extremes[self.column].values[0],
                        edgecolors='white', marker='o', facecolors='orangered', s=80, lw=1, zorder=20
                    )
                    point.set_visible(False)
                    annot.set_visible(False)

                    def update_annot(ind):
                        n = ind['ind'][0]
                        pos = points.get_offsets()[n]
                        annot.xy = pos
                        point.set_offsets(pos)
                        text = str(
                            f'Date : {self.extremes.index[n]}\n'
                            f'Value : {self.extremes[self.column].values[n]:.2f}\n'
                            f'Return Period : {return_periods[n]:.2f}'
                        )
                        annot.set_text(text)

                    def hover(event):
                        vis = annot.get_visible()
                        if event.inaxes == ax:
                            cont, ind = points.contains(event)
                            if cont:
                                update_annot(ind)
                                annot.set_visible(True)
                                point.set_visible(True)
                                fig.canvas.draw_idle()
                            else:
                                if vis:
                                    annot.set_visible(False)
                                    point.set_visible(False)
                                    fig.canvas.draw_idle()

                    fig.canvas.mpl_connect('motion_notify_event', hover)

                    fig.tight_layout()

                    return fig, ax

                # Probability plot
                else:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    points = ax.scatter(
                        theoretical, ecdf,
                        edgecolors='white', marker='o', facecolors='k', s=40, lw=1, zorder=10
                    )
                    lims = ax.get_xlim(), ax.get_ylim()
                    dlims = (-1e9, 1e9)
                    ax.plot(dlims, dlims, ls='--', lw=1, zorder=5, color='k')
                    ax.set_xlim(np.min(lims), np.max(lims))
                    ax.set_ylim(np.min(lims), np.max(lims))
                    ax.set_title(r'Probability Plot')
                    plt.xlabel(r'Theoretical probabilities')
                    plt.ylabel(rf'Observed probabilities, {plotting_position} plotting position')
                    ax.text(
                        .05, .9, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
                        s=f'$\\bar{{R}}^2$={r**2:>.2f}\np={p:>.3f}', fontsize=14,
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='k', lw=1, zorder=25)
                    )

                    annot = ax.annotate(
                        '', xy=(theoretical[0], self.extremes[self.column].values[0]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='k', lw=1, zorder=25),
                        zorder=30
                    )
                    point = ax.scatter(
                        theoretical[0], self.extremes[self.column].values[0],
                        edgecolors='white', marker='o', facecolors='orangered', s=80, lw=1, zorder=20
                    )
                    point.set_visible(False)
                    annot.set_visible(False)

                    def update_annot(ind):
                        n = ind['ind'][0]
                        pos = points.get_offsets()[n]
                        annot.xy = pos
                        point.set_offsets(pos)
                        text = str(
                            f'Date : {self.extremes.index[n]}\n'
                            f'Value : {self.extremes[self.column].values[n]:.2f}\n'
                            f'Return Period : {return_periods[n]:.2f}'
                        )
                        annot.set_text(text)

                    def hover(event):
                        vis = annot.get_visible()
                        if event.inaxes == ax:
                            cont, ind = points.contains(event)
                            if cont:
                                update_annot(ind)
                                annot.set_visible(True)
                                point.set_visible(True)
                                fig.canvas.draw_idle()
                            else:
                                if vis:
                                    annot.set_visible(False)
                                    point.set_visible(False)
                                    fig.canvas.draw_idle()

                    fig.canvas.mpl_connect('motion_notify_event', hover)

                    fig.tight_layout()

                    return fig, ax

        else:
            if quantiles:
                return (
                    (theoretical, self.extremes[self.column].values),
                    (r, p)
                )
            else:
                return (
                    (theoretical, ecdf),
                    (r, p)
                )

    def goodness_of_fit(self, method, **kwargs):
        """
        Calculates various goodness-of-fit statistics for selected model.

        Parameters
        ----------
        method : str
            Goodness of fit statistic method.
            Supported methods:
                'AIC' - Akaike information criterion
                    Lower value corresponds to a better fit.
                    see https://en.wikipedia.org/wiki/Akaike_information_criterion
                'log-likelihood' - log-likelihood
                    Higher value corresponds to a better fit.
                'KS' - Kolmogorov Smirnot test
                    Null hypothesis - both samples come from the same distribution.
                    If p<0.05 => reject Null hypothesis with p-level of confidence.
                    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
                'chi-square' - Chi-Square test
                    Null hypothesis - both samples come from the same distribution.
                    Calculates theoretical counts for given quantile ranges and compares to theoretical.
                    If p<0.05 => reject Null hypothesis with p-level of confidence.
                    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
        kwargs
            if fit is MCMC
                burn_in : int
                    Number of samples to discard. Samples, before the series converges, should be discarded.
                kernel_steps : int, optional
                    Number of bins (kernel support points) to determine mode (default=1000).
            for AIC
                order : int, optional
                    Order of AIC (1 for regular, 2 for small samples) (default=2).
                k : int
                    Number of parameters estimated by the model (fixed parameters don't count)
            fot KS
                mode : str, optional
                    See scipy docs (default='approx').
                alternative : str, optional
                    See scipy docs (default='two-sided').
            for chi-square
                chi_quantiles : int, optional
                    Number of equal slices (quantiles) into which observed data is split
                    to calculate the stitistic(default=4).
                k : int
                    Number of parameters estimated by the model (fixed parameters don't count)

        Returns
        -------
        if method = 'log-likelihood' : float, log-likelihood
        if method = 'AIC' : float, AIC statistic
        if method = 'KS' : tuple(statistic, p-value)
        if method = 'chi-square' : tuple(statistic, p-value)
        """

        # Make sure fit method was executed and fit data was generated
        if not self.__status['fit']:
            raise ValueError('No fit information found. Run self.fit() method first')

        if self.fit_method == 'MLE':
            fit_parameters = self.fit_parameters
        elif self.fit_method == 'MCMC':
            burn_in = kwargs.pop('burn_in')
            kernel_steps = kwargs.pop('kernel_steps', 1000)
            fit_parameters = self._kernel_fit_parameters(burn_in=burn_in, kernel_steps=kernel_steps)
        else:
            raise RuntimeError(f'Unexpected fit_method {self.fit_method}')

        distribution_object = getattr(scipy.stats, self.distribution_name)
        exceedances = self.extremes[self.column].values - self.threshold
        # Flip exceedances around 0
        if self.extremes_type == 'low':
            exceedances *= -1

        log_likelihood = np.sum(
            distribution_object.logpdf(exceedances, *fit_parameters)
        )

        if method == 'log-likelihood':
            assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'
            return log_likelihood

        elif method == 'AIC':
            order = kwargs.pop('order', 2)
            k = kwargs.pop('k')
            assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

            aic = 2 * k - 2 * log_likelihood

            if order == 1:
                return aic
            elif order == 2:
                return aic + (2 * k ** 2 + 2 * k) / (len(self.extremes) - k - 1)
            else:
                raise ValueError(f'order must be 1 or 2, {order} was passed')

        elif method == 'KS':
            mode = kwargs.pop('mode', 'approx')
            alternative = kwargs.pop('alternative', 'two-sided')
            assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

            exceedances = self.extremes[self.column].values - self.threshold
            if self.extremes_type == 'low':
                exceedances *= -1

            ks, p = scipy.stats.kstest(
                rvs=exceedances, cdf=distribution_object.cdf, args=fit_parameters,
                alternative=alternative, mode=mode
            )
            return ks, p

        elif method == 'chi-square':
            chi_quantiles = kwargs.pop('chi_quantiles', 4)
            k = kwargs.pop('k')
            assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'
            chi_quantile_ranges = [1 / chi_quantiles * (i + 1) for i in np.arange(-1, chi_quantiles)]
            observed_counts, expected_counts = [], []
            for i in range(chi_quantiles):
                bot = np.nanquantile(
                    self.extremes[self.column].values,
                    chi_quantile_ranges[i]
                )
                top = np.nanquantile(
                    self.extremes[self.column].values,
                    chi_quantile_ranges[i + 1]
                )
                if i + 1 == chi_quantiles:
                    observed_counts.append(
                        len(
                            self.extremes[
                                (self.extremes[self.column] >= bot)
                                & (self.extremes[self.column] <= top)
                            ]
                        )
                    )
                else:
                    observed_counts.append(
                        len(
                            self.extremes[
                                (self.extremes[self.column] >= bot)
                                & (self.extremes[self.column] < top)
                            ]
                        )
                    )
                expected_counts.append(
                    len(self.extremes) * (self.cdf(top) - self.cdf(bot))
                )
            if min(observed_counts) <= 5 or min(expected_counts) <= 5:
                raise ValueError(f'Too few observations in observed counts {min(observed_counts)} '
                                 f'or expected counts {min(expected_counts):.0f}, reduce chi_quantiles')
            cs, p = scipy.stats.chisquare(f_obs=observed_counts, f_exp=expected_counts, ddof=k)
            return cs, p

        else:
            raise ValueError(f'Method {method} not recognized')


if __name__ == "__main__":

    # Load data and initialize EVA
    import os
    df = pd.read_csv(
        os.path.join(os.getcwd(), r'test data\Battery_residuals.csv'),
        index_col=0, parse_dates=True
    )
    self = EVA(dataframe=df, column='Residuals (ft)', block_size=365.25, gap_length=24)

    # Set up test parameters
    etype = 'high'
    extremes_method = 'POT'
    _method = 'MCMC'
    mle_ci = 'Delta'

    if extremes_method == 'POT':
        _distribution = 'genpareto'
    elif extremes_method == 'BM':
        _distribution = 'genextreme'
    else:
        raise RuntimeError

    # Run a series of methods to assist in finding optimal threshold
    if extremes_method == 'POT':
        if etype == 'high':
            self.plot_mean_residual_life(
                thresholds=np.arange(2, 8, .01), r=24*7, alpha=.95,
                adjust_threshold=True, limit=10, extremes_type='high'
            )
            self.plot_parameter_stability(
                thresholds=np.arange(3, 8, .05), r=24*7, alpha=.95,
                adjust_threshold=True, limit=10, extremes_type='high'
            )
        elif etype == 'low':
            self.plot_mean_residual_life(
                thresholds=np.arange(-8, -2, .01), r=24*7, alpha=.95,
                adjust_threshold=True, limit=10, extremes_type='low'
            )
            self.plot_parameter_stability(
                thresholds=np.arange(-8, -2.5, .05), r=24*7, alpha=.95,
                adjust_threshold=True, limit=20, extremes_type='low'
            )

    # Extract extreme values
    if extremes_method == 'BM':
        self.get_extremes(method='BM', plotting_position='Weibull', extremes_type=etype)
    elif extremes_method == 'POT':
        if etype == 'high':
            self.get_extremes(method='POT', threshold=3, r=24*7, plotting_position='Weibull', extremes_type='high')
        elif etype == 'low':
            self.get_extremes(method='POT', threshold=-2.8, r=24*7, plotting_position='Weibull', extremes_type='low')
    self.plot_extremes()

    # Test independence of POT extremes
    if extremes_method == 'POT':
        self.test_extremes(method='autocorrelation')
        self.test_extremes(method='lag plot', lag=1)
        print(self.test_extremes(method='runs test', alpha=0.05))

    # Fit distribution
    if _method == 'MLE':
        if _distribution == 'genpareto':
            # Shape (f0) and location (floc) are both 0 => equivalent to exponential distribution (expon with floc=0)
            self.fit(distribution_name=_distribution, fit_method='MLE', scipy_fit_options=dict(floc=0))
        elif _distribution == 'genextreme':
            self.fit(distribution_name=_distribution, fit_method='MLE')
    elif _method == 'MCMC':
        self.fit(
            distribution_name=_distribution, fit_method='MCMC',
            nsamples=1000, nwalkers=200, starting_bubble=.01
        )
        # Trace plot
        if _distribution == 'genpareto':
            fig_trace, axes_trace = self.plot_trace(burn_in=200, labels=[r'$\xi$', r'$\sigma$'])
        elif _distribution == 'genextreme':
            fig_trace, axes_trace = self.plot_trace(burn_in=200, labels=[r'$\xi$', r'$\mu$', r'$\sigma$'])

        if _distribution == 'genpareto':
            fig_corner = self.plot_corner(burn_in=200, bins=50, labels=[r'$\xi$', r'$\sigma$'], smooth=1)
        elif _distribution == 'genextreme':
            fig_corner = self.plot_corner(burn_in=200, bins=50, labels=[r'$\xi$', r'$\mu$', r'$\sigma$'], smooth=1)

    # Test quality of fit
    if _method == 'MLE':
        print(self.goodness_of_fit(method='AIC', k=1))
        self.plot_qq(k=2, plotting_position='Weibull', quantiles=True)
        self.plot_qq(k=2, plotting_position='Weibull', quantiles=False)
    else:
        _burn_in = 200
        print(self.goodness_of_fit(method='AIC', k=2, burn_in=_burn_in, kernel_steps=100))
        self.plot_qq(
            k=2, plotting_position='Weibull', quantiles=True,
            rv_kwargs=dict(burn_in=_burn_in, estimate_method='parameter mode', kernel_steps=100)
        )
        self.plot_qq(
            k=2, plotting_position='Weibull', quantiles=False,
            rv_kwargs=dict(burn_in=_burn_in, estimate_method='parameter mode', kernel_steps=100)
        )

    # Generate results
    if _method == 'MCMC':
        _burn_in = 200
        self.generate_results(
            alpha=.95,
            rv_kwargs=dict(burn_in=_burn_in, estimate_method='parameter mode', kernel_steps=100),
            ci_kwargs=dict(burn_in=_burn_in)
        )
    elif _method == 'MLE':
        if mle_ci == 'Monte Carlo':
            self.generate_results(
                alpha=.95,
                ci_kwargs=dict(
                    method='Monte Carlo', k=100, source='data', sampling_method='constant', assume_normality=False
                )
            )
        elif mle_ci == 'Delta':
            self.generate_results(alpha=.95, ci_kwargs=dict(method='Delta', dx='1e-10', precision=100))

    # Plot extremes return plot
    if _method == 'MCMC':
        _burn_in = 200
        self.plot_summary(
            bins=10, plotting_position='Gringorten',
            rv_kwargs=dict(burn_in=200, estimate_method='parameter mode', kernel_steps=100)
        )
    elif _method == 'MLE':
        self.plot_summary(bins=10, plotting_position='Gringorten')

# coastlib, a coastal engineering Python library
# Copyright (C), 2019 Georgii Bocharov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import pickle
import warnings

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


class EVA:

    def __init__(self, data, block_size='1Y'):
        """
        Initialize an EVA class instance.

        Parameters
        ----------
        data : pandas.Series
            Pandas Series object containing data to be analyzed.
            Data must be numeric and index must be of type pandas.DatetimeIndex.
        block_size : str, optional
            Block size, used to determine number of blocks in data (default='1Y', one Gregorian year).
            See pandas.to_timedelta documentation for syntax.
            Block size is used to estimate probabilities (return periods for observed data) for all methods
            and to extract extreme events when using the 'Block Maxima' method.
            Return periods have units of <block_size> - e.g. a for block_size='1Y'
            a return period of 100 is the same thing as a 100-year return period.
            Weekly would be <block_size='1W'> and monthly would be approximately <block_size='30D'>.
        """

        if not isinstance(data, pd.Series):
            raise TypeError(f'<data> must be a {pd.Series} object, {type(data)} was received')

        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError(f'index of <data> must be a {pd.DatetimeIndex} object, {type(data.index)} was received')

        if not is_numeric_dtype(data):
            raise TypeError(f'<data> must be of numeric dtype, \'{data.dtype}\' was received')

        self.data = data.copy(deep=True)
        self.data.sort_index(ascending=True, inplace=True)
        nancount = self.data.isna().sum()
        if nancount > 0:
            self.data.dropna(inplace=True)
            warnings.warn(f'{nancount:d} no-data entries were dropped')

        self.__block_size = pd.to_timedelta(block_size)

        self.__extremes_method = None
        self.__extremes_type = None
        self.__threshold = None
        self.extremes = None

    @property
    def block_size(self):
        return self.__block_size

    @block_size.setter
    def block_size(self, value):
        """
        See <block_size> paramter in the __init__ method.
        """

        if not isinstance(value, str):
            raise TypeError(f'<block_size> must be {str}, {type(value)} was received')

        self.__block_size = pd.to_timedelta(value)

    @property
    def number_of_blocks(self):
        return (self.data.index[-1] - self.data.index[0]) / pd.to_timedelta(f'{self.block_size}D')

    @property
    def extremes_method(self):
        return self.__extremes_method

    @property
    def extremes_type(self):
        return self.__extremes_type

    @property
    def threshold(self):
        return self.__threshold

    def __repr__(self):
        series_length = (self.data.index[-1] - self.data.index[0]).total_seconds() / 60 / 60 / 24

        summary = [
            f'{" " * 35}Extreme Value Analysis Summary',
            f'{"=" * 100}',
            f'Analyzed parameter{self.data.name[:28]:>29}{" " * 6}Series length{series_length:29.2f} days',
        ]

        for i in range(1, int(np.ceil(len(self.data.name) / 28))):
            summary.append(f'{" " * 19}{self.data.name[i * 28:(i + 1) * 28]:<29}')

        summary.extend(
            [
                f'Block size{self.block_size:32.2f} days{" " * 6}Number of blocks{self.number_of_blocks:31.2f}',
                f'{"=" * 100}'
            ]
        )

        return '\n'.join(summary)

    def get_extremes(self, method='BM', plotting_position='Weibull', extremes_type='high', **kwargs):

        if method not in ['BM', 'POT']:
            raise ValueError(f'\'{method}\' is not a valid <method> value')
        self.__extremes_method = method

        if extremes_type not in ['high', 'low']:
            raise ValueError(f'\'{extremes_type}\' is not a valid <extremes_type> value')
        self.__extremes_type = extremes_type

        if method == 'BM':
            errors = kwargs.pop('errors', 'raise')
            if errors not in ['raise', 'coerce']:
                raise ValueError(f'\'{errors}\' is not a valid <errors> value')
            assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(", ".join(kwargs.keys()))

            self.__threshold = 0

            extremes_func = {
                'high': pd.Series.idxmax,
                'low': pd.Series.idxmin
            }[self.extremes_type]

            date_time_intervals = pd.interval_range(
                start=self.data.index[0], freq=self.block_size,
                periods=np.ceil(self.number_of_blocks), closed='left'
            )

            extreme_values, extreme_indices = [], []
            for interval in date_time_intervals:
                interval_slice = self.data.loc[
                    (self.data.index >= interval.left) &
                    (self.data.index < interval.right)
                ]
                try:
                    extreme_indices.append(extremes_func(interval_slice))
                    extreme_values.append(interval_slice.loc[extreme_indices[-1]])
                except ValueError as error_message:
                    if errors == 'coerce':
                        extreme_values.append(np.nan)
                        extreme_indices.append(interval.mid)
                    else:
                        raise ValueError(error_message)

        else:
            raise NotImplementedError

        self.extremes = pd.DataFrame(
            data=extreme_values,
            columns=['Return Value'],
            index=extreme_indices
        )
        self.extremes.index.name = self.data.index.name

        self.extremes.fillna(np.nanmean(extreme_values), inplace=True)

    def to_pickle(self, fname):
        with open(fname, 'wb') as output_stream:
            pickle.dump(self, output_stream)

    @classmethod
    def from_pickle(cls, fname):
        with open(fname, 'rb') as input_stream:
            return pickle.load(input_stream)


if __name__ == '__main__':
    import os
    ds = pd.read_csv(
        os.path.join(os.getcwd(), 'tests', '_common_data', 'wind_speed.csv'),
        index_col=0, parse_dates=True
    )['s'].rename('Wind Speed [kn]')
    self = EVA(data=ds.dropna(), block_size='30D')

    self.get_extremes(method='BM', plotting_position='Weibull', extremes_type='high', errors='coerce')

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

import time


def time2str(value):
    hours = value // 3600
    minutes = value % 3600 // 60
    seconds = value % 60
    ts = []
    for v in [hours, minutes]:
        if v < 10:
            ts.append(f'0{v:.0f}')
        else:
            ts.append(f'{v:.0f}')
    if seconds < 10:
        ts.append(f'0{seconds:.2f}')
    else:
        ts.append(f'{seconds:.2f}')
    return ':'.join(ts)


class ProgressBar:

    def __init__(self, total_iterations, bars=50, bar_items='0123456789#', prefix=''):
        """
        Progress bar.

        Parameters
        ----------
        total_iterations : int
            Number of iterations.
        bars : int, optional
            Length of progress bar (default=50).
        bar_items : str, optional
            String with characters used to represent progress bar elements. Last symbol represents a completed cell.
        prefix : str, optional
            Prefix placed before the progress bar (default=''). Used to describe progress of what is reported.
        """

        self.total_iterations = total_iterations
        self.bar_items = bar_items
        self.prefix = prefix

        self.__bars = bars

        self.__percent_per_bar = 100 / self.__bars
        self.__start_time = time.time()
        self.__i = 0

    @property
    def bars(self):
        return self.__bars

    @bars.setter
    def bars(self, value):
        if not isinstance(value, int):
            raise TypeError(f'value \'{value}\' is not valid for the <bars> parameter')
        self.__bars = value
        self.__percent_per_bar = 100 / self.__bars

    @property
    def i(self):
        return self.__i

    @i.setter
    def i(self, value):
        if value > self.total_iterations:
            raise ValueError(f'iterator value of \'{value}\' exceeds the iterator limit of \'{self.total_iterations}\'')
        if not isinstance(value, int) or value < 0:
            raise TypeError(f'\'{value}\' is not a valid iterator value')
        self.__i = value

    def __repr__(self):
        return self.progress_bar

    def increment(self, inc=1):
        self.i += inc

    @property
    def progress_bar(self):
        percentage = 100 * self.i / self.total_iterations
        full_bars = int(percentage / self.__percent_per_bar)
        bar = ['[']
        if full_bars > 0:
            bar.append(f'{self.bar_items[-1]}' * full_bars)
        if full_bars < self.bars:
            partial_bar_ind = int(
                percentage % self.__percent_per_bar / self.__percent_per_bar * (len(self.bar_items) - 1)
            )
            bar.append(self.bar_items[partial_bar_ind])
            bar.append(' ' * (self.bars - full_bars - 1))
        bar.append(']')
        bar = ''.join(bar)

        elapsed = time.time() - self.__start_time
        elapsed_string = time2str(elapsed)

        try:
            speed = self.i / elapsed
            eta = (self.total_iterations - self.i) / speed
            eta_string = time2str(eta)
        except ZeroDivisionError:
            speed = 0
            eta_string = f'00:00:00.00'

        return f'{self.prefix} {percentage:>3.0f}% ' \
               f'{bar} ' \
               f'{self.i:>{len(str(self.total_iterations))}}/' \
               f'{self.total_iterations:>{len(str(self.total_iterations))}} ' \
               f'[ETA: {eta_string}, Elapsed: {elapsed_string}, Speed: {speed:.2f} it/s]'

    def print(self):
        print(self.progress_bar, end='\r')

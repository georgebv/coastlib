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


def to_ds(value):
    if len(f'{value:.0f}') < 2:
        return f'0{value:.0f}'
    else:
        return f'{value:.0f}'


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
            String with characters used to represent intermediate states of bar. Last symbol represents a complete bar.
        prefix : str, optional
            Prefix placed before the progress bar (default='').
        """

        self.total_iterations = total_iterations
        self.bars = bars
        self.bar_items = bar_items
        self.prefix = prefix

        self.__start_time = time.time()
        self.__percent_per_bar = 100 / self.bars
        self.__i = 0
        self.__update_bar()

    @property
    def i(self):
        return self.__i

    @i.setter
    def i(self, value):
        if value > self.total_iterations:
            raise ValueError(f'Value {value} exceeds the iterator limit of {self.total_iterations}')
        if not isinstance(value, int) or value < 0:
            raise ValueError(f'{value} is not a valid iterator value')
        self.__i = value
        self.__update_bar()

    def __repr__(self):
        return self.progress_bar

    def increment(self, inc=1):
        self.i += inc

    def __update_bar(self):
        percentage = self.i / self.total_iterations * 100
        full_bars = int(percentage / self.__percent_per_bar)
        bar = '[' + f'{self.bar_items[-1]}' * full_bars
        if len(bar) < (self.bars + 1):
            partial_bar_ind = int(
                percentage % self.__percent_per_bar / self.__percent_per_bar * (len(self.bar_items) - 1)
            )
            bar += self.bar_items[partial_bar_ind]
        bar += ' ' * (self.bars - (len(bar) - 1)) + ']'

        elapsed = time.time() - self.__start_time
        elapsed_string = f'{to_ds(elapsed // 3600)}:' \
                         f'{to_ds(elapsed % 3600 // 60)}:' \
                         f'{to_ds(elapsed % 60)}.{str(elapsed % 1)[2]}'
        try:
            speed = self.i / elapsed
            eta = (self.total_iterations - self.i) / speed
            eta_string = f'{to_ds(eta // 3600)}:' \
                         f'{to_ds(eta % 3600 // 60)}:' \
                         f'{to_ds(eta % 60)}.{str(eta % 1)[2]}'
        except ZeroDivisionError:
            speed = 0
            eta_string = f'00:00:00.00'

        self.progress_bar = f'{self.prefix} {percentage:>3.0f}% ' \
                            f'{bar} ' \
                            f'{self.i:>{len(str(self.total_iterations))}}/' \
                            f'{self.total_iterations:>{len(str(self.total_iterations))}} ' \
                            f'[ETA: {eta_string}, Elapsed: {elapsed_string}, Speed: {speed:.2f} it/s]'

    def print(self):
        print(self.progress_bar, end='\r')

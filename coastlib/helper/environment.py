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

import coastlib.bin
import os


def append_bin():
    """
    Appends coastlib/bin to system environment PATH variable.
    Allows calling binary files from the terminal.
    """

    bin_path = os.path.split(str(coastlib.bin.__file__))[0]
    if bin_path not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep + bin_path

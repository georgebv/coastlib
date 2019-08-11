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

matplotlib_styles = {}

theme_color = '#454545'
coastlib_rc = {
    'font.size': 8,
    'font.style': 'normal',
    'font.weight': 'normal',
    'font.family': 'Arial',
    'text.color': theme_color,
    'axes.edgecolor': theme_color,
    'axes.labelcolor': theme_color,
    'axes.linewidth': .4,
    'axes.grid': False,
    'xtick.color': theme_color,
    'xtick.major.width': .4,
    'xtick.major.pad': 1,
    'xtick.minor.width': .4 * .75,
    'xtick.minor.visible': True,
    'xtick.bottom': False,
    'xtick.top': False,
    'ytick.color': theme_color,
    'ytick.major.width': .4,
    'ytick.major.pad': 1,
    'ytick.minor.width': .4 * .75,
    'ytick.minor.visible': True,
    'ytick.left': False,
    'grid.color': theme_color,
    'grid.linestyle': '-',
    'grid.linewidth': .5,
    'grid.alpha': .7,
    'legend.framealpha': 1,
    'legend.facecolor': 'w',
    'legend.edgecolor': theme_color,
    'legend.fontsize': 8,
    'patch.linewidth': .4
}
matplotlib_styles['coastlib_rc'] = coastlib_rc

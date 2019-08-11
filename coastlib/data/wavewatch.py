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

import cartopy.crs
import os
import numpy as np
import pandas as pd
import rasterio
import ftplib
import matplotlib.pyplot as plt
import datetime
import scipy.interpolate


class Open:
    """
    Downloads and parses GRIB (.grb or .grb2) data from NOAA WAVEWATCH III hindcast FTP server.
    Generates an object with attributes and methods described below.

    Parameters
    ----------
    full_path : str
        Path to grib file on NOAA ftp server. Other parameters (except save_folder) are ignored if full_path is provided
    group : str
        'nww3' for old (July 1999 to November 2007) and 'multi_1' for new (February 2005 till present)
    parameter : str
        Parameters to extract:
            wind    The 10 m wind speeds and direction gtom GFS
              hs    Significant wave height (in m)
              tp    Peak period (in s)
              dp    Average direction at the peak period
    year : int
        4-digit year value
    month : int
        Month value
    grid : str
        Grid name
        For multi_1:
            glo_30m     The 30 arc-minute global grid
             ao_30m     The 30 arc minute arctic grid
             ak_10m     The 10 arc minute regional Alaska grid
                        (it is 15 arc minute along longitude and 10 arc minute
                        along lattitude)
             at_10m     The 10 arc minute Northern Atlantic regional grid
                        (this grid covers the Gulf of Mexico and out to the
                        continental shelf on the US East Coast. It does not
                        extend across the Atlantic)
             wc_10m     The 10 arc minute regional grid for teh US West Coast
             ep_10m     The 10 arc minute grid covering some of the islands in
                        the Pacific (including the Hawaiian islands)
              ak_4m     The 4 arc minute coastal grid for Alaska
                        (8 arc minute along longituide and 4 arc minute along
                        lattitude)
              at_4m     The 4 arc minute coastal grid for the US East Coast
                        (includes the waters around the island of Peurto Rico)
              wc_4m     The 4 arc minute coastal grid for the US West Coast
                        (includes the coastal waters of Hawaii)
        For nww3:
                akw
                enp
                nah
                nph
               nww3
                wna
    save_folder : str, optional
        Path where grib files are saved (not saved by default)

    Attributes
    ----------
    self.path : str
        FTP link to the GRIB file
    self.data : np.ndarray
        1D array of 2D arrays representing grid with values, each 2D array represents a band (timestep/parameter)
    self.geotransform : np.ndarray
        Rasterio geotransform (dataset.affine)
    self.x_grid, self.y_grid : np.ndarray
        2D arrays representing x and y coordinate grids
    self.x, self.y : np.ndarray
        1D arrays representing x and y coordinates mapping self.data
    self.metadata : np.ndarray
        1D array of dicts with GRIB file metadata for each band represents a timestep/parameter
    self.ref_time : np.ndarray
        1D array with reference time for each band
    self.valid_time : np.ndarray
        1D array with valid time for each band
    self.parameters : np.ndarray
        1D array with parameter name for each band
    self.unique_parameters : np.ndarray
        1D array with unique parameter names
    self.unique_timestamps : np.ndarray
        1D array with unique timesteps

    Methods
    -------
    self.plot
        Generates a geographic raster map of selected parameter and timestep (see method docstring)
    self.point_series
        Generates a dataframe with interpolated timeseries for a given point
    self.line_series
        Generates a dataframe with interpolated timeseries for a given set of points
        or for an interval and number of points
    """

    def __init__(self, full_path=None, save_folder=None, **kwargs):
        if full_path is not None:
            self.path = full_path

        else:
            group = kwargs.pop('group')
            year = kwargs.pop('year')
            month = kwargs.pop('month')
            grid = kwargs.pop('grid')
            parameter = kwargs.pop('parameter')
            assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

            # Verify month value is valid
            if month in np.arange(1, 10):
                date = ''.join([str(year), '0', str(month)])
            elif month in np.arange(10, 13):
                date = ''.join([str(year), str(month)])
            else:
                raise ValueError(f'Month {month} is not valid')

            # Verify year value is valid
            if len(str(year)) != 4:
                raise ValueError(f'Year {year} is not valid')

            if group == 'nww3':
                # Verify dates are valid
                if int(date[:4]) > 2007 or (int(date[:4]) == 2007 and int(date[4:]) > 11):
                    raise ValueError(f'For dates starting February 2005 use multi-grid data (group=multi_1)')
                if int(date[:4]) < 1999 or (int(date[:4]) == 1999 and int(date[4:]) < 7):
                    raise ValueError(f'No data available before July 1999')

                # Construct ftp path
                ftp_folder = group
                filename = '.'.join([grid, parameter, date + '.grb'])

            elif group == 'multi_1':
                # Verify dates are valid
                if int(date[:4]) < 2005 or (int(date[:4]) == 2005 and int(date[4:]) < 2):
                    raise ValueError(f'For dates July 1999 to November 2007 use single grid data (group=nww3)')

                # Construct ftp path
                ftp_folder = '/'.join(
                    [
                        group,
                        date,
                        'gribs'
                    ]
                )
                filename = '.'.join([group, grid, parameter, date + '.grb2'])

            else:
                raise ValueError(f'Group {group} not recognized')

            self.path = '/'.join([r'ftp://polar.ncep.noaa.gov/pub/history/waves', ftp_folder, filename])

        self.__parse(path=self.path)

        if save_folder is not None:
            # Connect to FTP server
            ftp = ftplib.FTP('polar.ncep.noaa.gov')
            ftp.login()
            ftp.cwd('/'.join(self.path.split('/')[3:-1]))

            # Save grb file locally
            curdir = os.getcwd()
            os.chdir(save_folder)
            ftp.retrbinary('RETR ' + self.path.split('/')[-1], open(self.path.split('/')[-1], 'wb').write)
            os.chdir(curdir)

            # Close FTP connection
            ftp.close()

    def __parse(self, path):
        with rasterio.open(path) as f:
            # Read data as ndarray of shape (time, x, y) and replace no-value-cells with nans
            self.data = f.read().copy()
            self.data[self.data == f.nodata] = np.nan

            # Transform pixel indeces to coordinates of data CRS
            self.geotransform = f.affine
            x, y = np.empty(f.shape), np.empty(f.shape)
            for i, row in enumerate(np.arange(f.shape[0])):
                for j, col in enumerate(np.arange(f.shape[1])):
                    x[i][j], y[i][j] = self.geotransform * (col, row)
            self.x_grid, self.y_grid = x, y

            # Get only unique values of x and y coordinates for the data grid
            self.x, self.y = x[0], y.T[0]

            # Get metadata and time stamps for each band
            self.metadata = np.array([f.tags(i) for i in np.arange(1, f.count + 1)])
            parameters, ref_time, valid_time = [], [], []
            for row in self.metadata:
                ref_time.append(
                    datetime.datetime.utcfromtimestamp(
                        np.int64(
                            max(
                                row['GRIB_REF_TIME'].split(' '), key=len
                            )
                        )
                    )
                )
                valid_time.append(
                    datetime.datetime.utcfromtimestamp(
                        np.int64(
                            max(
                                row['GRIB_VALID_TIME'].split(' '), key=len
                            )
                        )
                    )
                )
                parameters.append(row['GRIB_COMMENT'])
            self.ref_time = np.array(ref_time)
            self.valid_time = np.array(valid_time)
            self.parameters = np.array(parameters)
            self.unique_parameters = np.unique(self.parameters)
            self.unique_timestamps = np.unique(self.valid_time)

    def __repr__(self):
        rer_string = f'\n' \
            f'NOAA WAVEWATCH III grid object\n' \
            f'------------------------------\n' \
            f'Data source  {self.path}\n' \
            f'Date range   {self.unique_timestamps[0]} - {self.unique_timestamps[-1]}\n' \
            f'Parameters   {self.unique_parameters}\n' \
            f'------------------------------'

        return rer_string

    def plot(self, **kwargs):
        """
        Generates a geographic raster map of selected parameter and timestep

        Parameters
        ----------
        kwargs
            parameter : str, optional
                Name of parameter (see self.unique_parameters)
            timestamp : datetime.datetime object, optional
                Timestamp value of parameter (see self.unique_timestamps)
            data_crs : cartopy.crs.* object, optional
                Projection of data default=cartopy.crs.PlateCarree()
            projection : cartopy.crs.* object, optional
                Projection of generated map default=cartopy.crs.Mollweide()
            subplots_kwargs : dict, optional
                Dictionary with keyword arguments passed to plt.subplots command
            pcolormesh_kwargs : dict, optional
                Dictionary with keyword arguments passed to plt.pcolormesh command
            colorbar_kwargs : dict, optional
                Dictionary with keyword arguments passed to plt.colorbar command
            coastline_kwargs : dict, optional
                Dictionary with keyword arguments passed to ax.coastlines command

        Returns
        -------
        fig, ax
        """

        data_crs = kwargs.pop('data_crs', cartopy.crs.PlateCarree())
        projection = kwargs.pop('projection', cartopy.crs.Mollweide(central_longitude=-80))
        parameter = kwargs.pop('parameter', self.unique_parameters[0])
        timestamp = kwargs.pop('timestamp', self.valid_time[0])
        subplots_kwargs = kwargs.pop('subplots_kwargs', dict())
        pcolormesh_kwargs = kwargs.pop('pcolormesh_kwargs', dict(cmap='viridis', shading='gourand'))
        colorbar_kwargs = kwargs.pop('colorbar_kwargs', dict())
        coastline_kwargs = kwargs.pop('coastline_kwargs', dict(color='black'))
        assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

        # Get data slice with <parameter> and <timestamp> to plot
        data = self.data[(self.valid_time == timestamp) & (self.parameters == parameter)]
        if len(data) == 1:
            data = data[0]
        else:
            raise RuntimeError

        # Produce the plot
        fig, ax = plt.subplots(**subplots_kwargs, subplot_kw=dict(projection=projection))
        pcmesh = ax.pcolormesh(
            self.x, self.y, data,
            transform=data_crs, **pcolormesh_kwargs
        )
        ax.coastlines(**coastline_kwargs)
        ax.add_feature(cartopy.feature.LAND)
        ax.set_global()
        ax.set_title(' - '.join([str(timestamp), parameter]))
        # cbar = plt.colorbar(pcmesh, **colorbar_kwargs)
        plt.colorbar(pcmesh, **colorbar_kwargs)

        return fig, ax

    def point_series(self, x0, y0, method='linear', **kwargs):
        """
        Generates a dataframe with interpolated timeseries for a given set of points
        or for an interval and number of points

        Parameters
        ----------
        x0 : float
            Longitude (-180 for West to +180 for East)
        y0 : float
            Latitude (-90 for South to +90 for North)
        method : str, optional
            Interpolation method (linear of nearest). (default=linear)
        kwargs
            parameter : str, optional
                Name of parameter (see self.unique_parameters).
                First (self.unique_parameters[0]) parameter by default.

        Returns
        -------
        pd.DataFrame
            Dataframe with datetime index and 1 column containing interpolated data
        """

        parameter = kwargs.pop('parameter', self.unique_parameters[0])
        assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

        # Transform x0 and y0 from geographic to index
        if x0 < 0:
            pts = (x0 + 360, y0)
        else:
            pts = (x0, y0)
        col, row = ~self.geotransform * pts

        # Get data slice with <parameter>
        data = self.data[self.parameters == parameter]

        # Interpolate parameters from regular grid
        interpolated_values = []
        for i, timestamp in enumerate(self.unique_timestamps):
            interpolating_function = scipy.interpolate.RegularGridInterpolator(
                points=[
                    np.arange(data[i].shape[0]),
                    np.arange(data[i].shape[1])
                ],
                values=data[i], method=method
            )
            interpolated_values.append(
                interpolating_function([row, col])[0]
            )

        # Check if values are valid
        if (~np.isnan(interpolated_values)).sum() == 0:
            raise ValueError(f'Point with coordinates {x0} {y0} is not valid for given grid (no data available)')

        return pd.DataFrame(data=interpolated_values, index=self.unique_timestamps, columns=[parameter])

    def line_series(self, x, y, intervals=None, method='linear', **kwargs):
        # TODO

        parameter = kwargs.pop('parameter', self.unique_parameters[0])
        assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

        if np.isscalar(x) and np.isscalar(y) and intervals is not None:
            # generate sequence of points
            raise NotImplementedError
        elif not np.isscalar(x) and not np.isscalar(y):
            # sequence of points
            points = np.array(
                [
                    (_x, _y) for _x, _y in zip(x, y)
                ]
            )
        else:
            raise ValueError('Provided input is not valid. Read method docstring')

        # Get coordinates for each point
        values = []
        for point in points:
            series = self.point_series(
                point[0], point[1], method=method, parameter=parameter
            )
            values.append(series[parameter].values)
        values = np.array(values)
        columns = ['Start'] + [f'{i+2}' for i in range(len(points)-2)] + ['End']

        return pd.DataFrame(data=values.T, index=self.unique_timestamps, columns=columns)


if __name__ == '__main__':
    self = Open(
        save_folder=None,
        group='multi_1', grid='glo_30m', parameter='hs', year=2012, month=10
    )
    _fig, _ax = self.plot(
        data_crs=cartopy.crs.PlateCarree(), timestamp=self.unique_timestamps[0],
        projection=cartopy.crs.Orthographic(central_longitude=-80, central_latitude=30),
        coastline_kwargs=dict(resolution='50m'),
        pcolormesh_kwargs=dict(cmap='viridis', shading='gourand', vmin=0, vmax=8)
    )
    _ax.scatter(-73.5, 40.5, transform=cartopy.crs.PlateCarree(), color='orangered')
    timeseries = self.point_series(x0=-73.5, y0=40.5, method='nearest')
    timeseries.plot()
    lineseries = self.line_series(
        x=np.linspace(-73.5, -73.5, 10),
        y=np.linspace(40.5, 30.5, 10), method='linear'
    )

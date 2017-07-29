import datetime
import os
import pandas as pd
import scipy.io
import numpy as np
import warnings


class SentinelV:
    """
    Sentinel V Workhorse ADCP data output class.
    """

    def __init__(self, path):
        self.mat = scipy.io.loadmat(path)
        self.waves = 'Execute the waves_parse method first'
        self.currents = 'Execute the currents_parse method first'
        self.model = 'Sentinel V Workhorse ADCP'
        dtype = self.mat['info']
        dindex = dtype.dtype
        dstructured = {n: dtype[n][0, 0] for n in dindex.names}
        data = {}
        for key in dstructured:
            data[key] = []
            for j in range(len(dstructured[key])):
                data[key] += [float(dstructured[key][j])]
        self.info = pd.DataFrame(data)

    def waves_parse(self):
        """
        Generates pandas dataframe from the matlab file with the waves data.
        """
        dtype = self.mat['waves']
        dindex = dtype.dtype
        dstructured = {n: dtype[n][0, 0] for n in dindex.names}
        # Generate data array
        data = {}
        for key in dstructured:
            data[key] = []
            for j in range(len(dstructured['time'])):
                data[key] += [float(dstructured[key][j])]
        del data['time']
        # Convert time to datetime
        time = []
        for i in range(len(dstructured['time'])):
            time += [datetime.datetime.fromtimestamp(dstructured['time'][i])]
        self.waves = pd.DataFrame(data, index=time)

    def currents_parse(self):
        """
        Generates pandas dataframe from the matlab file with the currents data.
        """
        dtype = self.mat['wt']
        dindex = dtype.dtype
        dstructured = {n: dtype[n][0, 0] for n in dindex.names}
        velocity = dstructured['vel']
        # Generate data array
        east = np.array([[depth[0] for depth in time] for time in velocity])
        north = np.array([[depth[1] for depth in time] for time in velocity])
        horizontal = (east ** 2 + north ** 2) ** 0.5

        up = np.array([[depth[2] for depth in time] for time in velocity])
        total = (horizontal ** 2 + up ** 2) ** 0.5

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            east_depth_averaged = np.array([np.nanmean(x) for x in east])
            north_depth_averaged = np.array([np.nanmean(x) for x in north])
            horizontal_depth_averaged = np.array([np.nanmean(x) for x in horizontal])

            east_depth_max = np.array([np.nanmax(x) for x in east])
            north_depth_max = np.array([np.nanmax(x) for x in north])
            horizontal_depth_max = np.array([np.nanmax(x) for x in horizontal])

        def angle(vector):
            if vector[0] > 0 and vector[1] > 0:
                # Q-I
                return np.rad2deg(np.arctan(np.abs(vector[0] / vector[1])))
            elif vector[0] > 0 and vector[1] == 0:
                return 90
            elif vector[0] > 0 and vector[1] < 0:
                # Q-II
                return 90 + np.rad2deg(np.arctan(np.abs(vector[1] / vector[0])))
            elif vector[0] == 0 and vector[1] < 0:
                return 180
            elif vector[0] < 0 and vector[1] < 0:
                # Q-III
                return 180 + np.rad2deg(np.arctan(np.abs(vector[0] / vector[1])))
            elif vector[0] < 0 and vector[1] == 0:
                return 270
            elif vector[0] < 0 and vector[1] > 0:
                return 270 + np.rad2deg(np.arctan(np.abs(vector[1] / vector[0])))
            elif vector[0] ==0 and vector[1] > 0:
                return 0
            else:
                return np.nan

        direction_depth_averaged = [
            angle((east_depth_averaged[i], north_depth_averaged[i])) for i in range(len(east_depth_averaged))
            ]
        direction_depth_max = [
            angle((east_depth_max[i], north_depth_max[i])) for i in range(len(east_depth_max))
            ]

        self.currents = pd.DataFrame(horizontal_depth_averaged, columns=['Depth averaged current speed [m/s]'])
        self.currents['Depth averaged current direction [deg N]'] = direction_depth_averaged
        self.currents['Depth maximized current speed [m/s]'] = horizontal_depth_max
        self.currents['Depth maximized current direction [deg N]'] = direction_depth_max

        def nanargmax(x):
            try:
                return np.nanargmax(x)
            except:
                return np.nan
        info = self.info.values
        self.currents['Peak location [m from seabed]'] = np.array([nanargmax(x) for x in horizontal]) * info[0][2] + info[0][3]

        # Get time from the 'waves' structure
        dtype = self.mat['waves']
        dindex = dtype.dtype
        dstructured = {n: dtype[n][0, 0] for n in dindex.names}
        time = []
        for i in range(len(dstructured['time'])):
            time += [datetime.datetime.fromtimestamp(dstructured['time'][i])]
        self.currents.index = time

        dtype = self.mat['sens']
        dindex = dtype.dtype
        dstructured = {n: dtype[n][0, 0] for n in dindex.names}
        self.currents['Pressure depths [m]'] = [x[0] for x in dstructured['pd']]


    def export(self, par, save_format='xlsx', save_name='data frame', save_path=None):

        if par == 'waves':
            df = self.waves
        elif par == 'currents':
            df = self.currents
        else:
            raise ImportError('ERROR: incorrect parameter entered. Use \'waves\' or \'currents\'')

        if save_path is None:
            full_path = os.getcwd() + '\\' + save_name
        else:
            full_path = save_path + '\\' + save_name

        if save_format == 'csv':
            df.to_csv(full_path + '.csv', sep=' ', na_rep='NaN')
        elif save_format == 'xlsx':
            df.to_excel(pd.ExcelWriter(full_path + '.xlsx'), na_rep='NaN', sheet_name='waves')

    def convert(self, par, *args, systems='m to ft'):
        """
        Converts selected values in dataframe *df* between metric and customary systems.

        Parameters
        ----------
        par : string
            Dataframe to be converted (i.e. 'waves')
        args : string
            Column labels in the dataframe (i.e. 'Hs')
        systems : string
            Conversion parameter ('m to ft' or 'ft to m')

        Returns
        -------
        converts data in the input dataframe
        """
        if par == 'waves':
            df = self.waves
        elif par == 'currents':
            df = self.currents
        else:
            raise ImportError('ERROR: incorrect parameter entered. Use \'waves\' or \'currents\'')

        if isinstance(df, str):
            raise ValueError('ERROR: Execute the _parse method first')
        else:
            for argument in args:
                if systems == 'm to ft':
                    df[argument] = df[argument].map(lambda x: x / 0.3048)
                elif systems == 'ft to m':
                    df[argument] = df[argument].map(lambda x: x * 0.3048)
                else:
                    raise ValueError('ERROR: inappropriate systems parameter')

        if par == 'waves':
            self.waves = df
        elif par == 'currents':
            self.currents = df

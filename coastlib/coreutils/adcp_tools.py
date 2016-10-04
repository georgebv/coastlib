import datetime
import functools

import pandas as pd
import scipy.io


class SentinelV:
    """
    Sentinel V Workhorse ADCP data output class.
    """

    def __init__(self, path):
        self.mat = scipy.io.loadmat(path)
        self.waves = 'Execute the waves_parse method first'
        self.currents = 'Execute the currents_parse method first'
        self.model = 'Sentinel V Workhorse ADCP'

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
            data[key] = [0] * len(dstructured['time'])
            for j in range(len(data[key])):
                data[key][j] = float(dstructured[key][j])
        del data['time']
        # Convert time to datetime
        time = [0] * len(dstructured['time'])
        for i in range(len(dstructured['time'])):
            time[i] = dstructured['time'][i]
            time[i] = datetime.datetime.fromtimestamp(time[i])
        self.waves = pd.DataFrame(data, index=time)

    def currents_parse(self):
        """
        Generates pandas dataframe from the matlab file with the currents data.
        """
        dtype = self.mat['wt']
        dindex = dtype.dtype
        dstructured = {n: dtype[n][0, 0] for n in dindex.names}
        # Generate data array
        data = {}

        # Get time from the 'waves' structure
        dtype = self.mat['waves']
        dindex = dtype.dtype
        dstructured = {n: dtype[n][0, 0] for n in dindex.names}
        time = [0] * len(dstructured['time'])
        for i in range(len(dstructured['time'])):
            time[i] = dstructured['time'][i]
            time[i] = datetime.datetime.fromtimestamp(time[i])
        self.currents = pd.DataFrame(data, index=time)

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


def splice(df_list, sort=True):
    """
    Takes a list of dataframes and returns a merged dataframe (sorted by default).
    """
    df = functools.reduce((lambda x, y: pd.concat([x, y])), df_list)
    if sort:
        df.sort_index(inplace=True)
    return df

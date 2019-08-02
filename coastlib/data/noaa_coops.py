from datetime import datetime
from urllib.request import urlopen
import json

from coastlib.helper.progress_bar import ProgressBar

import numpy as np
import pandas as pd


def nanfloat(v):
    if isinstance(v, (int, float)):
        return v
    elif isinstance(v, str) and len(v) > 0:
        try:
            return float(v)
        except ValueError:
            return v
    else:
        return np.nan


def coops_api(station, begin_date, end_date, product, datum, **kwargs):
    """
    Parses NOAA Tides & Currents web data via the NOAA CO-OPS data retrieval API.
    See https://tidesandcurrents.noaa.gov/api/ for reference.

    Parameters
    ----------
    station : int or str
        A 7 character station ID, or a currents station ID.
        Station listings for various products can be viewed at https://tidesandcurrents.noaa.gov
        or viewed on a map at Tides & Currents Station Map.
    begin_date : str
        Start date-time.
        yyyyMMdd, yyyyMMdd HH:mm, MM/dd/yyyy, or MM/dd/yyyy HH:mm.
    end_date : str
        End date-time.
        yyyyMMdd, yyyyMMdd HH:mm, MM/dd/yyyy, or MM/dd/yyyy HH:mm.
        Maximum retrieval time:
            31 days     All 6 minute data products
            1 year      Hourly Height, and High/Low
            10 years    Tide Predictions, Daily, and Monthly Means
    product : str
        water_level             Preliminary or verified water levels, depending on availability.
        air_temperature         Air temperature as measured at the station.
        water_temperature       Water temperature as measured at the station.
        wind                    Wind speed, direction, and gusts as measured at the station.
        air_pressure            Barometric pressure as measured at the station.
        air_gap                 Air Gap (distance between a bridge and the water's surface) at the station.
        conductivity            The water's conductivity as measured at the station.
        visibility              Visibility from the station's visibility sensor. A measure of atmospheric clarity.
        humidity                Relative humidity as measured at the station.
        salinity                Salinity and specific gravity data for the station.
        hourly_height           Verified hourly height water level data for the station.
        high_low                Verified high/low water level data for the station.
        daily_mean              Verified daily mean water level data for the station.
        monthly_mean            Verified monthly mean water level data for the station.
        one_minute_water_level  One minute water level data for the station.
        predictions             6 minute predictions water level data for the station.
        datums                  datums data for the stations.
        currents                Currents data for currents stations.
    datum : str
        CRD                     Columbia River Datum
        IGLD                    International Great Lakes Datum
        LWD                     Great Lakes Low Water Datum (Chart Datum)
        MHHW                    Mean Higher High Water
        MHW                     Mean High Water
        MTL                     Mean Tide Level
        MSL                     Mean Sea Level
        MLW                     Mean Low Water
        MLLW                    Mean Lower Low Water
        NAVD                    North American Vertical Datum
        STND                    Station Datum
    units : str, optional
        (default='metric')
        metric                  Metric (Celsius, meters, cm/s) units
        english                 English (fahrenheit, feet, knots) units
    time_zone : str, optional
        (default='gmt')
        gmt                     Greenwich Mean Time
        lst                     Local Standard Time. The time local to the requested station.
        lst_ldt                 Local Standard/Local Daylight Time. The time local to the requested station.
    interval : str, optional
        The interval for which Meteorological data is returned (default=None).
        The default is 6 minute interval and there is no need to specify it.
        The hourly interval is supported for Met data and Predictions data only.
            h                   Hourly Met data and predictions data will be returned
            hilo                High/Low tide predictions for subordinate stations.
    bin : int, optional
        # TODO - this option is not yet implemented
        Currents data for bin number <bin> of the specified station is returned.
        If a bin is not specified for a PORTS station, the data is returned using a predefined real-time bin.

    Returns
    -------
    df : pd.DataFrame
        Pandas DataFrame with parsed time series.

    Raises
    ------
    ValueError
        if data is not available
    RuntimeError
        if data type is not recognized

    Examples
    --------
    >>> df = coops_api(station=8518750, begin_date='20121025', end_date='20121101', product='water_level', datum='NAVD')
    >>> df.columns.to_list()
    ['v', 's', 'f', 'q']
    >>> df['v'].mean()
    1.5010661458333334
    """

    units = kwargs.pop('units', 'english')
    time_zone = kwargs.pop('time_zone', 'gmt')
    interval = kwargs.pop('interval', None)
    assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

    link = rf'https://tidesandcurrents.noaa.gov/api/datagetter?product={product}' \
           rf'&application=coastlib' \
           rf'&begin_date={begin_date}' \
           rf'&end_date={end_date}' \
           rf'&datum={datum}' \
           rf'&station={station}' \
           rf'&time_zone={time_zone}' \
           rf'&units={units}' \
           rf'&format=json'
    if interval is not None:
        link += rf'&interval={interval}'

    with urlopen(link) as link_data:
        raw_data = json.loads(link_data.read().decode())

    if 'error' in raw_data.keys():
        raise ValueError(
                f'NOAA CO-OPS API ERROR: {raw_data["error"]["message"]}\n'
                f'Link {link}.\n'
            )

    if 'metadata' in raw_data.keys():
        data_key = list(raw_data.keys())[1]
        value_keys = list(raw_data[data_key][0].keys())
        columns = value_keys[1:]
        index = [
            datetime.strptime(row[value_keys[0]], '%Y-%m-%d %H:%M')
            for row in raw_data[data_key]
            ]
        data = [
            [nanfloat(row[key]) for key in columns]
            for row in raw_data[data_key]
        ]
        df = pd.DataFrame(
            data=data,
            index=index,
            columns=columns
        )
        df.index.name = f'Time ({time_zone})'
        return df

    elif 'predictions' in raw_data.keys():
        data_key = list(raw_data.keys())[0]
        value_keys = list(raw_data[data_key][0].keys())
        columns = value_keys[1:]
        index = [
            datetime.strptime(row[value_keys[0]], '%Y-%m-%d %H:%M')
            for row in raw_data[data_key]
        ]
        if len(columns) > 1:
            data = [
                [
                    nanfloat(row[key])
                    for key in columns
                ] for row in raw_data[data_key]
            ]
        else:
            data = [nanfloat(row[columns[0]]) for row in raw_data[data_key]]
        df = pd.DataFrame(
            data=data,
            index=index,
            columns=columns
        )
        df.index.name = f'Time ({time_zone})'
        return df

    elif 'datums' in raw_data.keys():
        index = [row['n'] for row in raw_data['datums']]
        data = [nanfloat(row['v']) for row in raw_data['datums']]
        df = pd.DataFrame(
            data=data,
            index=index,
            columns=['Value']
        )
        df.index.name = 'Datum'
        return df

    else:
        raise RuntimeError


def coops_api_batch(begin_date, end_date, return_logs=False, echo_progress=False, **kwargs):
    """
    Expands functionality of the <coops_api> function allowing to extract data for
    arbitrary time periods.

    Parameters
    ----------
    begin_date : datetime.datetime or str
        Start date-time - must be either a datetime object or a string convertible to datetime object.
    end_date : datetime.datetime or str
        End date-time - must be either a datetime object or a string convertible to datetime object.
    return_logs : bool, optional
        If True, returns a dictionary with logs for all iterations.
    echo_progress : bool, optional
        If True, prints out progress bar (default=False).
    kwargs
        See <coops_api> docstring for a full list of arguments.

    Returns
    -------
    df : pd.DataFrame
        Pandas DataFrame with parsed time series.

    Examples
    --------
    >>> df = coops_api_batch(station=8518750, begin_date='20120825', end_date='20121101', product='water_level', datum='NAVD')
    >>> df.columns.to_list()
    ['v', 's', 'f', 'q']
    >>> df['v'].mean()
    0.29498921634703756
    """

    product = kwargs['product']
    if product in ['daily_mean', 'monthly_mean']:
        time_delta = pd.to_timedelta('3500D')
    elif product in ['hourly_height', 'high_low']:
        time_delta = pd.to_timedelta('350D')
    else:
        time_delta = pd.to_timedelta('30D')

    if not isinstance(begin_date, datetime):
        begin_date = pd.to_datetime(begin_date)

    if not isinstance(end_date, datetime):
        end_date = pd.to_datetime(end_date)

    logs = {}
    progress_bar = ProgressBar(
        total_iterations=int(np.ceil((end_date - begin_date) / time_delta)),
        bars=50, bar_items='0123456789#', prefix='NOAA CO-OPS'
    )
    if echo_progress:
        print(progress_bar, end='\r')

    data = []
    _start = begin_date
    _end = begin_date + time_delta
    while _end - time_delta <= end_date:
        try:
            data.append(coops_api(begin_date=_start.strftime('%Y%m%d'), end_date=_end.strftime('%Y%m%d'), **kwargs))
            logs[len(logs)] = {
                'start': begin_date,
                'end': end_date,
                'success': True,
                'error_message': None
            }
        except ValueError as _err:
            logs[len(logs)] = {
                'start': begin_date,
                'end': end_date,
                'success': False,
                'error_message': str(_err)
            }
        _start += time_delta
        _end += time_delta
        if echo_progress:
            progress_bar.increment()
            print(progress_bar, end='\r')

    if len(data) > 0:
        df = pd.concat(data)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df = df[(df.index >= begin_date) & (df.index <= end_date)]
    else:
        df = None
    if echo_progress:
        print(progress_bar)

    if return_logs:
        return df, logs

    return df


def coops_datum(station, units='metric', metadata=False):
    """
    Parses NOAA Tides & Currents tidal datums for given station via the NOAA CO-OPS data retrieval API.

    Parameters
    ----------
    station : int or str
        A 7 character station ID, or a currents station ID.
        Station listings for various products can be viewed at https://tidesandcurrents.noaa.gov
        or viewed on a map at Tides & Currents Station Map.
    units : str, optional
        (default='metric')
        metric                  Metric (Celsius, meters, cm/s) units
        english                 English (fahrenheit, feet, knots) units
    metadata : bool, optional
        If True, also returns metadata (default=False).

    Returns
    -------
    df : pd.DataFrame
        Pandas DataFrame with parsed station datum.
    md : pd.DataFrame, optional
        Pandas DataFrame with parsed station datum metadata.

    Raises
    ------
    ValueError
        if data is not available

    Examples
    --------
    >>> datum = coops_datum(station=8518750)
    >>> datum.loc['MHHW'].values[0]
    2.543
    """

    link = f'https://tidesandcurrents.noaa.gov/mdapi/latest/webapi/stations/{station}' \
           f'/datums.json?units={units}&format=json'

    with urlopen(link) as link_data:
        raw_data = json.loads(link_data.read().decode())

    if 'error' in raw_data.keys():
        raise ValueError(
                f'NOAA CO-OPS API ERROR: {raw_data["error"]["message"]}\n'
                f'Link {link}.\n'
            )

    index = [row['name'] for row in raw_data['datums']]
    columns = ['Value', 'Description']
    data = [
        [nanfloat(row[key.lower()]) for key in columns]
        for row in raw_data['datums']
    ]
    df = pd.DataFrame(
        data=data,
        index=index,
        columns=columns
    )
    df.index.name = 'Datum'

    if metadata:
        index, data = [], []
        for key in raw_data.keys():
            if key not in ['datums', 'disclaimers', 'self']:
                index.append(key)
                data.append(raw_data[key])
        md = pd.DataFrame(
            data=data,
            index=index,
            columns=['Value']
        )
        md.index.name = 'Property'
        return df, md

    return df


if __name__ == "__main__":
    _df, _logs = coops_api_batch(
        station=8518750, begin_date='18120125', end_date='18121101', product='water_level',
        datum='NAVD', echo_progress=True, return_logs=True
    )
    # coops_api_data = coops_api(station=8518750, begin_date='20121025', end_date='20121101', product='water_level')
    # coops_api_data.plot()

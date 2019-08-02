from datetime import datetime
from urllib.request import urlopen
import json

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
    See https://tidesandcurrents.noaa.gov/api/

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
    >>> df['v'].max()
    11.28
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
    coops_api_data = coops_api(station=8518750, begin_date='20121025', end_date='20121101', product='water_level')
    coops_api_data.plot()

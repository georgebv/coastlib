from datetime import datetime
from urllib.request import urlopen

import numpy as np
import pandas as pd


def nanfloat(v):
    try:
        return float(v)
    except ValueError:
        return np.nan


def coops(station, begin_date, end_date, **kwargs):
    """
    Parses NOAA Tides & Currents web data via the CO-OPS API.
    See https://tidesandcurrents.noaa.gov/api/

    Parameters
    ----------
    station : int or str
        A 7 character station ID or a currents station ID.
    begin_date : str
        yyyyMMdd, yyyyMMdd HH:mm, MM/dd/yyyy, or MM/dd/yyyy HH:mm.
    end_date : str
        yyyyMMdd, yyyyMMdd HH:mm, MM/dd/yyyy, or MM/dd/yyyy HH:mm.
    kwargs
        product : str, optional (default='predictions')
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
        application : str, optional (default='web_services')
            web_services    The internal application Web Services has called the API.
            NDBC            A user from the National Data Buoy Center has called the API.
        interval : str, optional (default=None)
            h       Hourly Met data and predictions data will be returned.
            hilo    High/Low tide predictions for subordinate stations.
        datum : str, optional (default='NAVD')
            CRD         Columbia River Datum
            IGLD        International Great Lakes Datum
            LWD         Great Lakes Low Water Datum (Chart Datum)
            MHHW        Mean Higher High Water
            MHW         Mean High Water
            MTL         Mean Tide Level
            MSL         Mean Sea Level
            MLW         Mean Low Water
            MLLW        Mean Lower Low Water
            NAVD        North American Vertical Datum
            STND        Station Datum
        time_zone : str, optional (default='gmt')
            gmt         Greenwich Mean Time.
            lst         Local Standard Time. The time local to the requested station.
            lst_ldt     Local Standard/Local Daylight Time. The time local to the requested station.
        units : str, optional (default='english')
            metric      Metric (Celsius, meters, cm/s) units.
            english     English (fahrenheit, feet, knots) units.

    Returns
    -------
    df : pd.DataFrame
        Pandas DataFrame with parsed time series.
    """

    product = kwargs.pop('product', 'predictions')
    application = kwargs.pop('application', 'web_services')
    datum = kwargs.pop('datum', 'NAVD')
    time_zone = kwargs.pop('time_zone', 'gmt')
    units = kwargs.pop('units', 'english')
    interval = kwargs.pop('interval', None)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    link = str(
        rf'https://tidesandcurrents.noaa.gov/api/datagetter?product={product}'
        rf'&application={application}'
        rf'&begin_date={begin_date}'
        rf'&end_date={end_date}'
        rf'&datum={datum}'
        rf'&station={station}'
        rf'&time_zone={time_zone}'
        rf'&units={units}'
        rf'&format=csv'
    )
    if interval:
        link += rf'&interval={interval}'

    file = urlopen(link).read().decode()
    if file.split('\n')[1][:5] == 'Error':
        _err_msg = file.split('\n')[1]
        raise ValueError(
            f'{_err_msg}\n'
            f'No data available at {link}.\n'
            f'Either request parameters are incorrect, or there is no data for provided time range and station.'
        )
    else:
        columns = np.array(file.split('\n')[0].split(', '), dtype='U50')

        # Rename duplicate columns
        for item in columns:
            if (columns == item).sum() > 1:
                columns[columns == item] = np.array([
                    f'{_item}_{i}'
                    for i, _item in zip(range((columns == item).sum()), columns[columns == item])
                ], dtype='U50')

        data_raw = np.array([v for v in file.split('\n')[1:] if len(v) > 0])
        data = []
        for i in range(len(columns) - 1):
            data.append(
                [
                    nanfloat(v.split(',')[i + 1])
                    for v in data_raw
                ]
            )
        try:
            df = pd.DataFrame(
                data=np.transpose(data),
                index=[datetime.strptime(v.split(',')[0], '%Y-%m-%d %H:%M') for v in data_raw],
                columns=columns[1:]
            )
            df.index.name = f'Time ({time_zone})'
        except ValueError:
            df = pd.DataFrame(
                data=np.transpose(data),
                index=[v.split(',')[0] for v in data_raw],
                columns=columns[1:]
            )
        return df


if __name__ == "__main__":
    coops_data = coops(8518750, '20180101', '20180201')
    coops_data.plot()

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

from datetime import datetime

import pandas as pd
import requests
from coastlib.helper import ProgressBar


def ncei_datasets(
        start_date=None, end_date=None, bounding_box=(90, -180, -90, 180),
        keywords=None, text=None, available=True
):
    """
    Retrieve NOAA NCEI global formats, datasets, and datatypes
    https://www.ncei.noaa.gov/support/access-search-service-api-user-documentation

    Parameters
    ----------
    start_date : str, optional
        Start date in the ISO 8601 format (default=None).
            YYYY-MM-DD
                examples:
                    1776-07-04
                    1941-12-07
            YYYY-MM-DDTHH:mm:ss
                with Z or z for UTC
                and +HH:mm or -HH:mm for the offset from UTC
                examples:
                    2001-11-02T12:45:00Z
                    2001-11-02T12:45:00z
                    2001-11-02T08:45:00+04:00
    end_date : str, optional
        Start date in the ISO 8601 format (default=None).
        See the `start_date` argument for format description.
    bounding_box : array-like, optional
        The bounding box is used to select data from a geographic location contained within the coordinates,
        given as four comma separated numbers. North and South range from -90 to 90 and East and West range
        from -180 to 180. If these are not set the geographic extent defaults to the entire globe (90,-180,-90,180).
    keywords : tuple or str, optional
        A comma separated list of Global Change Master Directory (GCMD)
        (https://earthdata.nasa.gov/earth-observation-data/find-data/gcmd)
        Keywords or other terms that can be used to locate datasets.
        The keywords that are available come from the ISO 19115-2 metadata records.
        A full list is available from the service in an empty query in the keywords.bucket array.
        The complete list of GCMD keywords are available as an XML document here
        (https://gcmdservices.gsfc.nasa.gov/kms/concepts/concept_scheme/sciencekeywords)
        Here is an example of the keyword parameter.
            keywords='precipitation'
            keywords=('precipitation', 'ocean winds')
    text : str, optional
        The text parameter used to locate datasets by matching the terms of the dataset name and description.
        (default=None)
            text='rain'
    available : bool, optional
        A boolean (True or False) used to locate datasets based on
        whether they are available in the Data Search Endpoint (default=True).

    Returns
    -------
    formats : list
        List of available formats.
    datasets : list
        List of available datasets.
    datatypes : dict
        For each dataset gives a dictionary with names: id's

    Raises
    ------
    ValueError
        if REST response status code is 400 or 500
    """

    request_endpoint = r'https://www.ncei.noaa.gov/access/services/search/v1/datasets?'

    request_arguments = []

    assert isinstance(start_date, type(end_date)),\
        f'`start_date` and `end_date` are both required if one of them is passed'
    if start_date is not None:
        request_arguments.append(f'startDate={start_date}')
    if end_date is not None:
        request_arguments.append(f'endDate={end_date}')

    if len(bounding_box) != 4:
        raise TypeError(f'Invalid `bounding_box` passed in \'{bounding_box}\'')
    request_arguments.append(f'boundingBox={",".join([str(v) for v in bounding_box])}')

    if keywords is not None:
        if isinstance(keywords, str):
            request_arguments.append(f'keywords={keywords}')
        else:
            request_arguments.append(f'keywords={",".join(str(v) for v in keywords)}')

    if text is not None:
        request_arguments.append(f'text={text}')

    request_arguments.append(f'available={str(available).lower()}')

    request_url = request_endpoint + '&'.join(request_arguments)

    response = requests.get(request_url)
    data = response.json()

    if response.status_code == 400:
        raise ValueError(
            f'\n'
            f'    errorMessage: {data["errorMessage"]}\n'
            f'    errorCode: {data["errorCode"]}\n'
            f'    errors: {data["errors"]}'
        )
    elif response.status_code == 500:
        raise ValueError(
            f'\n'
            f'    errorMessage: {data["errorMessage"]}\n'
            f'    errorCode: {data["errorCode"]}'
        )

    formats = [ds['key'] for ds in data['formats']['buckets']]
    datasets = [ds['key'] for ds in data['datasets']['buckets']]
    datatypes = {
       ds['id']: {dt['name']: dt['id'] for dt in ds['dataTypes']}
       for ds in data['results']
    }

    return formats, datasets, datatypes


def ncei_search(
        dataset, start_date=None, end_date=None, bounding_box=(90, -180, -90, 180),
        datatypes=None, stations=None, limit=None, offset=None
):
    """
    Retrieve NOAA NCEI formats, datasets, and datatypes for given parameters
    https://www.ncei.noaa.gov/support/access-search-service-api-user-documentation

    !!! ATTENTION !!!
    This method is not documented by NOAA and was inferred from provided examples and by messing with the API.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g. 'local-climatological-data').
        See `ncei_datasets()` method for available dataset names.
    start_date : str, optional
        Start date in the ISO 8601 format (default=None).
            YYYY-MM-DD
                examples:
                    1776-07-04
                    1941-12-07
            YYYY-MM-DDTHH:mm:ss
                with Z or z for UTC
                and +HH:mm or -HH:mm for the offset from UTC
                examples:
                    2001-11-02T12:45:00Z
                    2001-11-02T12:45:00z
                    2001-11-02T08:45:00+04:00
    end_date : str, optional
        Start date in the ISO 8601 format (default=None).
        See the `start_date` argument for format description.
    bounding_box : array-like, optional
        The bounding box is used to select data from a geographic location contained within the coordinates,
        given as four comma separated numbers. North and South range from -90 to 90 and East and West range
        from -180 to 180. If these are not set the geographic extent defaults to the entire globe (90,-180,-90,180).
    datatypes : str or array-like, optional
        Accepts a valid data type id or a list of data type ids.
        Data returned will contain all of the data type(s) specified.
    stations : str or array-like, optional
        Accepts a valid station id or a list of of station ids.
        Data returned will contain data for the station(s) specified.
    limit : int, optional
        Defaults to 25, limits the number of results in the response. Maximum is 1000.
    offset : int, optional
        Defaults to 0, used to offset the resultlist.

    Returns
    -------
    datatypes
        List of available datasets for all stations.
    stations
        List of available station id's.
    results
        For each dataset gives a dictionary with
            id: {name: '', datatypes: []}

    Raises
    ------
    ValueError
        if REST response status code is 400 or 500
    """

    request_endpoint = r'https://www.ncei.noaa.gov/access/services/search/v1/data?'

    request_arguments = [f'dataset={dataset}']

    assert isinstance(start_date, type(end_date)),\
        f'`start_date` and `end_date` are both required if one of them is passed'
    if start_date is not None:
        request_arguments.append(f'startDate={start_date}')
    if end_date is not None:
        request_arguments.append(f'endDate={end_date}')

    if len(bounding_box) != 4:
        raise TypeError(f'Invalid `bounding_box` passed in \'{bounding_box}\'')
    request_arguments.append(f'boundingBox={",".join([str(v) for v in bounding_box])}')

    if datatypes is not None:
        if isinstance(datatypes, str):
            request_arguments.append(f'dataTypes={datatypes}')
        else:
            request_arguments.append(f'dataTypes={",".join(str(v) for v in datatypes)}')

    if stations is not None:
        if isinstance(stations, str):
            request_arguments.append(f'stations={stations}')
        else:
            request_arguments.append(f'stations={",".join(str(v) for v in stations)}')

    if limit is not None:
        request_arguments.append(f'limit={limit:d}')

    if offset is not None:
        request_arguments.append(f'offset={offset:d}')

    request_url = request_endpoint + '&'.join(request_arguments)

    response = requests.get(request_url)
    data = response.json()

    if response.status_code == 400:
        raise ValueError(
            f'\n'
            f'    errorMessage: {data["errorMessage"]}\n'
            f'    errorCode: {data["errorCode"]}\n'
            f'    errors: {data["errors"]}'
        )
    elif response.status_code == 500:
        raise ValueError(
            f'\n'
            f'    errorMessage: {data["errorMessage"]}\n'
            f'    errorCode: {data["errorCode"]}'
        )

    datatypes = [ds['key'] for ds in data['dataTypes']['buckets']]
    stations = [ds['key'] for ds in data['stations']['buckets']]
    results = {
        ds['stations'][0]['id']: {
            'name': ds['stations'][0]['name'],
            'datatypes': {dt['name']: dt['id'] for dt in ds['dataTypes']}
        } for ds in data['results']
    }

    return datatypes, stations, results


def ncei_api(
        dataset, stations, start_date=None, end_date=None,
        datatypes=None, bounding_box=(90, -180, -90, 180), output_format='json', units='metric'
):
    """
    Retrieve data from the NOAA NCEI data service
    https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation

    Parameters
    ----------
    dataset : str
        Dataset name (e.g. 'local-climatological-data').
        See `ncei_datasets()` method for available dataset names.
    stations : str or array-like
        Accepts a valid station id or a list of of station ids.
        Data returned will contain data for the station(s) specified.
    start_date : str, optional
        Start date in the ISO 8601 format (default=None).
            YYYY-MM-DD
                examples:
                    1776-07-04
                    1941-12-07
            YYYY-MM-DDTHH:mm:ss
                with Z or z for UTC
                and +HH:mm or -HH:mm for the offset from UTC
                examples:
                    2001-11-02T12:45:00Z
                    2001-11-02T12:45:00z
                    2001-11-02T08:45:00+04:00
    end_date : str, optional
        Start date in the ISO 8601 format (default=None).
        See the `start_date` argument for format description.
    datatypes : str or array-like, optional
        Accepts a valid data type id or a list of data type ids.
        Data returned will contain all of the data type(s) specified.
    bounding_box : array-like, optional
        The bounding box is used to select data from a geographic location contained within the coordinates,
        given as four comma separated numbers. North and South range from -90 to 90 and East and West range
        from -180 to 180. If these are not set the geographic extent defaults to the entire globe (90,-180,-90,180).
    output_format : str, optional
        (default='json')
        The format parameter allows the user to select how the data should be formatted.
        Note that some data formats ignore certain data types.
        For example, PDF data may only display data types for a report, and not for the requested dataTypes.
        These are the available formats:
            csv — Comma-Separated Values (CSV)
            ssv — Space-Separated Values (SSV)
            json — JavaScript Object Notation (JSON)
            pdf — Portable Document Format (PDF)
            netcdf  — Network Common Data Form (NetCDF)
    units : str, optional
        The units parameter converts the output data for datasets and datatypes
        that support conversion to either “metric” or “standard” units (default='metric').

    Returns
    -------
    if `output_format` is not 'json'
        content : bytes
            response.content (see `requests` module)
    else
        data : pandas.DataFrame
            DataFrame with parsed output with multiindex in the form ['STATION', 'DATE'].

    Raises
    ------
    ValueError
        if REST response status code is 400 or 500
    """

    request_endpoint = r'https://www.ncei.noaa.gov/access/services/data/v1?'

    request_arguments = [f'dataset={dataset}']

    if stations is not None:
        if isinstance(stations, str):
            request_arguments.append(f'stations={stations}')
        else:
            request_arguments.append(f'stations={",".join(str(v) for v in stations)}')

    assert isinstance(start_date, type(end_date)),\
        f'`start_date` and `end_date` are both required if one of them is passed'
    if start_date is not None:
        request_arguments.append(f'startDate={start_date}')
    if end_date is not None:
        request_arguments.append(f'endDate={end_date}')

    if datatypes is not None:
        if isinstance(datatypes, str):
            request_arguments.append(f'dataTypes={datatypes}')
        else:
            request_arguments.append(f'dataTypes={",".join(str(v) for v in datatypes)}')

    if len(bounding_box) != 4:
        raise TypeError(f'Invalid `bounding_box` passed in \'{bounding_box}\'')
    request_arguments.append(f'boundingBox={",".join([str(v) for v in bounding_box])}')

    request_arguments.append(f'format={output_format}')

    request_arguments.append('includeAttributes=true')
    request_arguments.append('includeStationName=true')
    request_arguments.append('includeStationLocation=1')

    request_arguments.append(f'units={units}')

    request_url = request_endpoint + '&'.join(request_arguments)

    response = requests.get(request_url)

    if response.status_code == 400:
        raise ValueError(
            f'\n'
            f'    errorMessage: {response.json()["errorMessage"]}\n'
            f'    errorCode: {response.json()["errorCode"]}\n'
            f'    errors: {response.json()["errors"]}'
        )
    elif response.status_code == 500:
        raise ValueError(
            f'\n'
            f'    errorMessage: {response.json()["errorMessage"]}\n'
            f'    errorCode: {response.json()["errorCode"]}'
        )

    if output_format != 'json':
        return response.content

    data = response.json()

    df = pd.DataFrame(data)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['STATION'] = df['STATION'] + ' : ' + df['NAME']
    del df['NAME']
    df.set_index(['STATION', 'DATE'], inplace=True)

    return df


def ncei_api_batch(
        dataset, stations, start_date=None, end_date=None, time_delta='350D', return_logs=False, echo_progress=False,
        datatypes=None, bounding_box=(90, -180, -90, 180), units='metric'
):
    """
    Expands functionality of the <ncei_api> function allowing extraction of data for
    arbitrary time periods.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g. 'local-climatological-data').
        See `ncei_datasets()` method for available dataset names.
    stations : str or array-like
        Accepts a valid station id or a list of of station ids.
        Data returned will contain data for the station(s) specified.
    start_date : str, optional
        Start date - must be either a datetime object or a string convertible to datetime object.
    end_date : str, optional
        Start date - must be either a datetime object or a string convertible to datetime object.
    time_delta : str or pandas.Timedelta, optional
        Interval of data collection (default='350D'). If string, must be convertible to pandas.Timedelta.
    return_logs : bool, optional
        If True, returns a dictionary with logs for all iterations.
    echo_progress : bool, optional
        If True, prints out progress bar (default=False).
    datatypes : str or array-like, optional
        Accepts a valid data type id or a list of data type ids.
        Data returned will contain all of the data type(s) specified.
    bounding_box : array-like, optional
        The bounding box is used to select data from a geographic location contained within the coordinates,
        given as four comma separated numbers. North and South range from -90 to 90 and East and West range
        from -180 to 180. If these are not set the geographic extent defaults to the entire globe (90,-180,-90,180).
    units : str, optional
        The units parameter converts the output data for datasets and datatypes
        that support conversion to either “metric” or “standard” units (default='metric').

    Returns
    -------
    if return_logs :
        df : pd.DataFrame
            Pandas DataFrame with parsed time series.
        logs : dict
    df : pd.DataFrame
        Pandas DataFrame with parsed time series.

    Raises
    ------
    ValueError
        if REST response status code is 400 or 500
    """

    if not isinstance(time_delta, pd.Timedelta):
        time_delta = pd.to_timedelta(time_delta)

    if not isinstance(start_date, datetime):
        start_date = pd.to_datetime(start_date)

    if not isinstance(end_date, datetime):
        end_date = pd.to_datetime(end_date)

    logs = {}
    progress_bar = ProgressBar(
        total_iterations=int((end_date - start_date) / time_delta) + 1,
        bars=50, bar_items='0123456789#', prefix='NOAA NCEI'
    )
    if echo_progress:
        print(progress_bar, end='\r')

    data = []
    _start = start_date
    _end = start_date + time_delta
    while _end - time_delta <= end_date:
        try:
            data.append(
                ncei_api(
                    dataset=dataset, stations=stations,
                    start_date=_start.strftime('%Y-%m-%dT%H:%M:%S'), end_date=_end.strftime('%Y-%m-%dT%H:%M:%S'),
                    datatypes=datatypes, bounding_box=bounding_box, output_format='json', units=units
                )
            )
            logs[len(logs)] = {
                'start': _start,
                'end': _end,
                'success': True,
                'error_message': None
            }
        except ValueError as _err:
            logs[len(logs)] = {
                'start': _start,
                'end': _end,
                'success': False,
                'error_message': str(_err)
            }
        except KeyError:
            logs[len(logs)] = {
                'start': _start,
                'end': _end,
                'success': False,
                'error_message': 'Retrieved data has non-standard format. Try using the <ncei_api> function.'
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
    else:
        df = None
    if echo_progress:
        print(progress_bar)

    if return_logs:
        return df, logs

    return df


if __name__ == '__main__':
    nd_data = ncei_datasets()
    for key, value in nd_data[-1].items():
        print(f'{key} :\n    {", ".join(value.keys())}')

    ns_data = ncei_search(dataset='local-climatological-data', stations='72503014732', limit=5)
    for key, value in ns_data[-1].items():
        print(f'{key} -> {value["name"]}\n')
    for key, value in ns_data[-1]['72503014732']['datatypes'].items():
        if 'dir' in key.lower() or 'speed' in key.lower():
            print(f'{key} -> {value}')

    na_data = ncei_api(
        dataset='local-climatological-data', stations=['72503014732', '70219026615'],
        start_date='2012-10-01', end_date='2012-11-01'
    )

    b_data = ncei_api_batch(
        dataset='local-climatological-data', stations=['72503014732', '70219026615'],
        start_date='2010-01-01', end_date='2011-01-01', echo_progress=True, time_delta='10D'
    )

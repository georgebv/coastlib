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

import requests


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
            f'errorMessage: {data["errorMessage"]}\n'
            f'errorCode: {data["errorCode"]}\n'
            f'errors: {data["errors"]}'
        )
    elif response.status_code == 500:
        raise ValueError(
            f'errorMessage: {data["errorMessage"]}\n'
            f'errorCode: {data["errorCode"]}'
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
    offset int, optional
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
            f'errorMessage: {data["errorMessage"]}\n'
            f'errorCode: {data["errorCode"]}\n'
            f'errors: {data["errors"]}'
        )
    elif response.status_code == 500:
        raise ValueError(
            f'errorMessage: {data["errorMessage"]}\n'
            f'errorCode: {data["errorCode"]}'
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


def ncei_api():
    """
    Retrieve data from the NOAA NCEI data service
    https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation

    Returns
    -------

    """

    request_endpoint = r'https://www.ncei.noaa.gov/access/services/data/v1?'


if __name__ == '__main__':
    nd_data = ncei_datasets()
    for key, value in nd_data[-1].items():
        print(f'{key} :\n    {", ".join(value.keys())}')

    ns_data = ncei_search(dataset='local-climatological-data', stations='72503014732')
    for key, value in ns_data[-1].items():
        print(f'{key} -> {value["name"]}\n')
    for key, value in ns_data[-1]['72503014732']['datatypes'].items():
        if 'dir' in key.lower() or 'speed' in key.lower():
            print(f'{key} -> {value}')

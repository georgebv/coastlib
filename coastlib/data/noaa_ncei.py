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

    request_url = '&'.join([request_endpoint, *request_arguments])

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


def ncei_api():
    request_endpoint = r'https://www.ncei.noaa.gov/access/services/data/v1'


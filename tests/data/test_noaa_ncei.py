import pytest
import numpy as np
import pandas as pd
from coastlib.data.noaa_ncei import ncei_datasets, ncei_search, ncei_api, ncei_api_batch


def test_ncei_datasets():
    datasets = ncei_datasets(
        start_date='20080101', end_date='20120101', bounding_box=(90, -180, -90, 180),
        keywords=None, text='wind', available=True
    )

    for fmt in ['csv', 'json', 'pdf']:
        assert fmt in datasets[0]

    for ds in ['global-hourly', 'global-marine', 'global-summary-of-the-day', 'local-climatological-data']:
        assert ds in datasets[1]

    assert 'local-climatological-data' in datasets[2].keys()

    assert 'Hourly Wind Speed' in datasets[2]['local-climatological-data'].keys()


def test_ncei_search():
    with pytest.raises(ValueError):
        data = ncei_search(
            dataset='local-climatological-data', stations=['72503014732', '70219026615'], limit=5,
            start_date='20080101', end_date='20120101', bounding_box=(90, -180, -90, 180), offset=0,
            datatypes=['HourlyWindSpeed', 'HourlyWindDirection']
        )

    data = ncei_search(dataset='local-climatological-data', stations=['72775024143', '72503014732'], limit=1000)

    assert 'HourlyWindSpeed' in data[0]

    for stn in ['72775024143', '72503014732']:
        assert stn in data[1]

    for stn in ['72503014732', '72775024143']:
        assert stn in data[2].keys()
    assert data[2]['72503014732']['name'] == 'LA GUARDIA AIRPORT, NY US'
    assert 'Hourly Wind Speed' in data[2]['72503014732']['datatypes'].keys()


def test_ncei_api():
    data = ncei_api(
        dataset='local-climatological-data', stations=['72503014732', '70219026615'],
        start_date='2012-10-01', end_date='2012-11-01', datatypes=['HourlyWindSpeed', 'HourlyWindDirection']
    )

    assert data.index.levels[1][0].year == 2012
    assert data.index.levels[1][-1].year == 2012

    assert data.index.levels[1][0].month == 10
    assert data.index.levels[1][-1].month == 11

    assert np.all(
        data.index.levels[0].values == [
            '70219026615 : BETHEL AIRPORT, AK US', '72503014732 : LA GUARDIA AIRPORT, NY US'
        ]
    )

    for clmn in ['HourlyWindSpeed', 'HourlyWindDirection']:
        assert clmn in data.columns

    assert int(pd.to_numeric(data['HourlyWindSpeed'], errors='coerce').mean()) == 12
    assert int(pd.to_numeric(data['HourlyWindDirection'], errors='coerce').mean()) == 157


def test_ncei_api_batch():
    data = ncei_api_batch(
        dataset='local-climatological-data', stations='72503014732',
        start_date='2010-01-01', end_date='2011-01-01', time_delta='1Y',
        datatypes=['HourlyWindSpeed', 'HourlyWindDirection']
    )

    assert data.index.levels[1][0].year == 2010
    assert data.index.levels[1][-1].year == 2011

    assert data.index.levels[1][0].month == 1
    assert data.index.levels[1][-1].month == 1

    for clmn in ['HourlyWindSpeed', 'HourlyWindDirection']:
        assert clmn in data.columns

    assert int(pd.to_numeric(data['HourlyWindSpeed'], errors='coerce').mean()) == 11
    assert int(pd.to_numeric(data['HourlyWindDirection'], errors='coerce').mean()) == 201

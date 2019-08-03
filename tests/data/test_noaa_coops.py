from coastlib.data.noaa_coops import coops_api, coops_api_batch, coops_datum
import numpy as np
import sys
import io


def test_coops_api_basic():
    df = coops_api(station=8518750, begin_date='20121025', end_date='20121101', product='water_level', datum='NAVD')
    assert len(df) == 1920
    assert np.allclose(df.mean().values, np.array([1.50106615, 0.10458333]))


def test_coops_api_custom():
    df = coops_api(
        station=8518750, begin_date='20121025', end_date='20121101', product='predictions', datum='MLLW',
        units='english', time_zone='lst'
    )
    assert len(df) == 1920
    assert np.allclose(df.mean().values, np.array([2.6943510]))


def test_coops_api_datum():
    df = coops_api(
        station=8518750, begin_date='20121025', end_date='20121101', product='datums', datum='MLLW',
        units='english', time_zone='lst'
    )
    assert len(df) == 14
    assert np.allclose(df.mean().values, np.array([5.9407142]))


def test_coops_api_batch_basic():
    df = coops_api_batch(
        station=8518750, begin_date='20120825', end_date='20121001', product='water_level', datum='NAVD'
    )
    assert len(df) == 8881
    assert np.allclose(df.mean().values, np.array([0.15666164, 0.12670837]))


def test_coops_api_batch_custom():
    stdout_ = sys.stdout
    stream = io.StringIO()
    sys.stdout = stream
    df, logs = coops_api_batch(
        station=8518750, begin_date='20130101', end_date='20130307', product='water_level', datum='NAVD',
        return_logs=True, echo_progress=True
    )
    sys.stdout = stdout_
    assert stream.getvalue()[:80] == 'NOAA CO-OPS   0% [0                                                 ] 0/3 [ETA: '
    assert stream.getvalue()[-120:-60] == ' 100% [##################################################] 3'

    assert len(df) == 15601
    assert np.allclose(df.mean().values, np.array([-0.33339863,  0.1075654]))
    assert len(logs) == 3
    assert str(logs[1]['end']) == '2013-03-02 00:00:00'


def test_coops_datum_basic():
    datum = coops_datum(station=8518750)
    assert len(datum) == 15
    assert np.allclose(datum.mean().values, np.array([1.69]))


def test_coops_datum_metadata():
    datum, meta = coops_datum(station=9416841, units='english', metadata=True)
    assert len(datum) == 15
    assert np.allclose(datum.mean().values, np.array([18.14733333]))
    assert len(meta) == 20
    assert str(meta.loc['ctrlStation'].values) == "['9415020 Point Reyes, CA']"

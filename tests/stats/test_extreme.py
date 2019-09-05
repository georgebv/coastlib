import pytest
import pandas as pd
import numpy as np
from coastlib.stats.extreme import EVA
import pathlib


data_path = pathlib.Path(__file__).parent.parent / '_common_data'


@pytest.fixture(scope='module')
def data_raw():
    return pd.read_csv(data_path / 'wind_speed.csv', index_col=0, parse_dates=True)['s'].rename('Wind Speed [kn]')


@pytest.fixture(scope='module')
def data():
    return pd.read_csv(
        data_path / 'wind_speed.csv', index_col=0, parse_dates=True
    )['s'].rename('Wind Speed [kn]').dropna()


class TestEVA:

    def test_init(self, data_raw, data):
        # must be pandas.Series
        with pytest.raises(TypeError):
            _ = EVA(data=pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 1]}))

        # must have date-time index
        with pytest.raises(TypeError):
            _ = EVA(data=pd.Series({'a': 1, 'b': 2}))

        # must be numeric
        with pytest.raises(TypeError):
            _ = EVA(data=pd.Series({pd.to_datetime('06/01/2000'): 'a', pd.to_datetime('06/02/2000'): 'b'}))

        # nan-values removed
        with pytest.warns(UserWarning):
            _ = EVA(data=data_raw)

    def test_default_block_size(self, data):
        eva = EVA(data=data)
        assert eva.block_size == 365.2425
        assert np.isclose(eva.number_of_blocks, 9.99883638952203)

    def test_custom_block_size(self, data):
        eva = EVA(data=data, block_size=90)
        assert eva.block_size == 90
        assert np.isclose(eva.number_of_blocks, 40.577777777777776)

"""Common fixtures for all tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from unittest.mock import Mock

from seapopym.configuration.no_transport.forcing_parameter import ChunkParameter, ForcingUnit
from seapopym.standard.coordinates import new_latitude, new_longitude, new_time, new_layer


@pytest.fixture
def sample_coordinates():
    """Create sample standardized coordinates for testing."""
    time_data = pd.date_range('2023-01-01', periods=12, freq='ME')
    lat_data = np.arange(-90, 91, 10)
    lon_data = np.arange(-180, 180, 20)
    layer_data = [0, -50, -100]

    return {
        'time': new_time(time_data),
        'latitude': new_latitude(lat_data),
        'longitude': new_longitude(lon_data),
        'layer': new_layer(layer_data)
    }


@pytest.fixture
def sample_dataarray_standardized(sample_coordinates):
    """Create a sample DataArray with standardized coordinates."""
    coords = sample_coordinates
    data = np.random.random((
        coords['time'].size,
        coords['latitude'].size,
        coords['longitude'].size
    ))

    return xr.DataArray(
        data,
        coords={
            'T': coords['time'],
            'Y': coords['latitude'],
            'X': coords['longitude']
        },
        dims=['T', 'Y', 'X'],
        attrs={'units': 'degC', 'standard_name': 'sea_surface_temperature'}
    )


@pytest.fixture
def sample_dataarray_old_coords():
    """Create a sample DataArray with old coordinate names."""
    time_data = pd.date_range('2023-01-01', periods=12, freq='ME')
    lat_data = np.arange(-90, 91, 10)
    lon_data = np.arange(-180, 180, 20)

    data = np.random.random((len(time_data), len(lat_data), len(lon_data)))

    return xr.DataArray(
        data,
        coords={
            'time': ('time', time_data, {'standard_name': 'time', 'axis': 'T'}),
            'latitude': ('latitude', lat_data, {'standard_name': 'latitude', 'axis': 'Y', 'units': 'degrees_north'}),
            'longitude': ('longitude', lon_data, {'standard_name': 'longitude', 'axis': 'X', 'units': 'degrees_east'})
        },
        dims=['time', 'latitude', 'longitude'],
        attrs={'units': 'degC', 'standard_name': 'sea_surface_temperature'}
    )


@pytest.fixture
def sample_forcing_unit_standardized(sample_dataarray_standardized):
    """Create a ForcingUnit with standardized coordinates."""
    return ForcingUnit(forcing=sample_dataarray_standardized)


@pytest.fixture
def sample_forcing_unit_old_coords(sample_dataarray_old_coords):
    """Create a ForcingUnit with old coordinates (will be auto-standardized)."""
    return ForcingUnit(forcing=sample_dataarray_old_coords)


@pytest.fixture
def sample_chunk_parameter():
    """Create a sample ChunkParameter with standardized attributes."""
    return ChunkParameter(
        functional_group=2,
        Y=18,
        X=36
    )


@pytest.fixture
def sample_chunk_dict():
    """Create expected chunk dictionary with standardized coordinates."""
    return {
        'functional_group': 2,
        'Y': 18,
        'X': 36,
        'T': -1
    }


# Phase 2 fixtures removed - using static class analysis instead of instances
# to avoid complex validation requirements in protocol tests


@pytest.fixture
def mock_dask_client():
    """Mock Dask client for testing distributed functionality."""
    mock_client = Mock()
    mock_client.scatter.return_value = Mock()
    return mock_client


@pytest.fixture
def sample_dataset_coherent(sample_coordinates):
    """Create a dataset with coherent coordinates for validation testing."""
    coords = sample_coordinates

    # Temperature data
    temp_data = np.random.random((
        coords['time'].size,
        coords['latitude'].size,
        coords['longitude'].size
    )) * 30

    # Primary production data (same coordinates)
    pp_data = np.random.random((
        coords['time'].size,
        coords['latitude'].size,
        coords['longitude'].size
    )) * 100

    return xr.Dataset({
        'temperature': (['T', 'Y', 'X'], temp_data, {'units': 'degC'}),
        'primary_production': (['T', 'Y', 'X'], pp_data, {'units': 'mg/m3/day'})
    }, coords={
        'T': coords['time'],
        'Y': coords['latitude'],
        'X': coords['longitude']
    })


@pytest.fixture
def sample_dataset_incoherent():
    """Create a dataset with incoherent coordinates for validation testing."""
    # Different time coordinates
    time1 = pd.date_range('2023-01-01', periods=12, freq='ME')
    time2 = pd.date_range('2023-02-01', periods=10, freq='ME')  # Different

    lat_data = np.arange(-90, 91, 10)
    lon_data = np.arange(-180, 180, 20)

    temp_coords = {
        'T': new_time(time1),
        'Y': new_latitude(lat_data),
        'X': new_longitude(lon_data)
    }

    pp_coords = {
        'T': new_time(time2),  # Incoherent time
        'Y': new_latitude(lat_data),
        'X': new_longitude(lon_data)
    }

    temp_data = np.random.random((len(time1), len(lat_data), len(lon_data))) * 30
    pp_data = np.random.random((len(time2), len(lat_data), len(lon_data))) * 100

    temp_da = xr.DataArray(temp_data, coords=temp_coords, dims=['T', 'Y', 'X'])
    pp_da = xr.DataArray(pp_data, coords=pp_coords, dims=['T', 'Y', 'X'])

    return {
        'temperature': ForcingUnit(forcing=temp_da),
        'primary_production': ForcingUnit(forcing=pp_da)
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Any global setup can go here
    np.random.seed(42)  # For reproducible tests
    yield
    # Cleanup after test if needed
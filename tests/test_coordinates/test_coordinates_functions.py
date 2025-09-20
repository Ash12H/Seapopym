"""Tests for coordinates.py functions that create standardized coordinates."""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

from seapopym.standard.coordinates import new_latitude, new_longitude, new_time, new_layer, new_cohort


@pytest.mark.coordinates
class TestCoordinateFunctions:
    """Test class for coordinate creation functions."""

    def test_new_latitude_creates_standardized_coordinate(self):
        """Test that new_latitude() creates a coordinate named Y."""
        lat_data = np.array([0, 10, 20])
        lat_coord = new_latitude(lat_data)

        assert lat_coord.name == 'Y'
        assert lat_coord.dims == ('Y',)
        np.testing.assert_array_equal(lat_coord.values, lat_data)

        # Check CF attributes
        assert lat_coord.attrs['standard_name'] == 'latitude'
        assert lat_coord.attrs['units'] == 'degrees_north'
        assert lat_coord.attrs['axis'] == 'Y'

    def test_new_longitude_creates_standardized_coordinate(self):
        """Test that new_longitude() creates a coordinate named X."""
        lon_data = np.array([-10, 0, 10])
        lon_coord = new_longitude(lon_data)

        assert lon_coord.name == 'X'
        assert lon_coord.dims == ('X',)
        np.testing.assert_array_equal(lon_coord.values, lon_data)

        # Check CF attributes
        assert lon_coord.attrs['standard_name'] == 'longitude'
        assert lon_coord.attrs['units'] == 'degrees_east'
        assert lon_coord.attrs['axis'] == 'X'

    def test_new_time_creates_standardized_coordinate(self):
        """Test that new_time() creates a coordinate named T."""
        time_data = pd.date_range('2023-01-01', periods=5, freq='D')
        time_coord = new_time(time_data)

        assert time_coord.name == 'T'
        assert time_coord.dims == ('T',)
        np.testing.assert_array_equal(time_coord.values, time_data.values)

        # Check CF attributes
        assert time_coord.attrs['standard_name'] == 'time'
        assert time_coord.attrs['axis'] == 'T'

    def test_new_layer_creates_standardized_coordinate(self):
        """Test that new_layer() creates a coordinate named Z."""
        layer_data = [0, -50, -100]
        layer_coord = new_layer(layer_data)

        assert layer_coord.name == 'Z'
        assert layer_coord.dims == ('Z',)
        np.testing.assert_array_equal(layer_coord.values, layer_data)

        # Check CF attributes
        assert layer_coord.attrs['standard_name'] == 'layer'
        assert layer_coord.attrs['axis'] == 'Z'
        assert layer_coord.attrs['positive'] == 'down'

    def test_new_layer_default_values(self):
        """Test new_layer() with default depth values."""
        layer_coord = new_layer()  # No data provided, should use defaults

        assert layer_coord.name == 'Z'
        assert layer_coord.dims == ('Z',)
        assert len(layer_coord.values) > 0  # Should have default values

    def test_new_cohort_creates_coordinate(self):
        """Test that new_cohort() creates a cohort coordinate."""
        cohort_data = [0, 1, 2, 3]
        cohort_coord = new_cohort(cohort_data)

        assert cohort_coord.name == 'cohort'
        assert cohort_coord.dims == ('cohort',)
        np.testing.assert_array_equal(cohort_coord.values, cohort_data)

        # Check attributes
        assert cohort_coord.attrs['standard_name'] == 'cohort'

    def test_cf_xarray_detection_on_created_coordinates(self):
        """Test that cf_xarray can detect created coordinates."""
        lat_coord = new_latitude(np.array([0, 10, 20]))
        lon_coord = new_longitude(np.array([-10, 0, 10]))
        time_coord = new_time(pd.date_range('2023-01-01', periods=3))

        # Create a simple DataArray with these coordinates
        data = np.random.random((3, 3, 3))
        da = xr.DataArray(
            data,
            coords={'T': time_coord, 'Y': lat_coord, 'X': lon_coord},
            dims=['T', 'Y', 'X']
        )

        # Test cf_xarray detection
        assert da.cf['T'].name == 'T'
        assert da.cf['Y'].name == 'Y'
        assert da.cf['X'].name == 'X'

    def test_coordinate_functions_return_proper_types(self):
        """Test that coordinate functions return xr.DataArray coordinates."""
        lat_coord = new_latitude(np.array([0, 10]))
        lon_coord = new_longitude(np.array([0, 10]))
        time_coord = new_time(pd.date_range('2023-01-01', periods=2))
        layer_coord = new_layer([0, -50])

        # All should be DataArray coordinates
        assert isinstance(lat_coord, xr.DataArray)
        assert isinstance(lon_coord, xr.DataArray)
        assert isinstance(time_coord, xr.DataArray)
        assert isinstance(layer_coord, xr.DataArray)

        # All should be 1D
        assert lat_coord.ndim == 1
        assert lon_coord.ndim == 1
        assert time_coord.ndim == 1
        assert layer_coord.ndim == 1

    def test_coordinates_can_be_used_in_dataset_creation(self):
        """Test that created coordinates can be used to build datasets."""
        lat_coord = new_latitude(np.array([0, 10]))
        lon_coord = new_longitude(np.array([-5, 5]))
        time_coord = new_time(pd.date_range('2023-01-01', periods=2))

        # Should be able to create a dataset
        data = np.random.random((2, 2, 2))
        ds = xr.Dataset(
            data_vars={
                'temperature': (['T', 'Y', 'X'], data)
            },
            coords={
                'T': time_coord,
                'Y': lat_coord,
                'X': lon_coord
            }
        )

        assert 'T' in ds.coords
        assert 'Y' in ds.coords
        assert 'X' in ds.coords
        assert ds.temperature.dims == ('T', 'Y', 'X')

    @pytest.mark.parametrize("lat_values", [
        np.array([0]),
        np.array([-90, 0, 90]),
        np.arange(-90, 91, 30),
        np.linspace(-90, 90, 10)
    ])
    def test_new_latitude_various_inputs(self, lat_values):
        """Test new_latitude with various input arrays."""
        lat_coord = new_latitude(lat_values)

        assert lat_coord.name == 'Y'
        assert lat_coord.dims == ('Y',)
        assert len(lat_coord) == len(lat_values)
        np.testing.assert_array_equal(lat_coord.values, lat_values)

    @pytest.mark.parametrize("lon_values", [
        np.array([0]),
        np.array([-180, 0, 180]),
        np.arange(-180, 181, 60),
        list(range(-180, 181, 90))  # Test with list input
    ])
    def test_new_longitude_various_inputs(self, lon_values):
        """Test new_longitude with various input arrays."""
        lon_coord = new_longitude(lon_values)

        assert lon_coord.name == 'X'
        assert lon_coord.dims == ('X',)
        assert len(lon_coord) == len(lon_values)
        np.testing.assert_array_equal(lon_coord.values, lon_values)

    def test_time_coordinates_with_different_frequencies(self):
        """Test new_time with different time frequencies."""
        # Daily frequency
        daily = pd.date_range('2023-01-01', periods=7, freq='D')
        daily_coord = new_time(daily)
        assert daily_coord.name == 'T'
        assert len(daily_coord) == 7

        # Monthly frequency
        monthly = pd.date_range('2023-01-01', periods=12, freq='ME')
        monthly_coord = new_time(monthly)
        assert monthly_coord.name == 'T'
        assert len(monthly_coord) == 12

    def test_fixtures_integration(self, sample_coordinates):
        """Test integration with conftest.py fixtures."""
        coords = sample_coordinates

        # All should be properly named
        assert coords['time'].name == 'T'
        assert coords['latitude'].name == 'Y'
        assert coords['longitude'].name == 'X'
        assert coords['layer'].name == 'Z'

        # All should be DataArrays
        assert all(isinstance(coord, xr.DataArray) for coord in coords.values())
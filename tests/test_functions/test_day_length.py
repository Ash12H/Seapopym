"""Tests for day_length function with standardized coordinates."""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

from seapopym.function.day_length import _mesh_day_length
from seapopym.standard.coordinates import new_latitude, new_longitude, new_time


@pytest.mark.scientific
class TestDayLength:
    """Test class for day length calculation functions."""

    def test_mesh_day_length_with_standardized_coordinates(self):
        """Test _mesh_day_length with standardized T/Y/X coordinates."""
        # Create standardized coordinates
        time_data = pd.date_range('2023-01-01', periods=4, freq='ME')
        lat_data = np.array([0, 30, 60])  # Different latitudes
        lon_data = np.array([0, 90])  # Different longitudes

        time = new_time(time_data)
        latitude = new_latitude(lat_data)
        longitude = new_longitude(lon_data)

        # Test the function
        result = _mesh_day_length(time, latitude, longitude, angle_horizon_sun=0)

        # Check result structure
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('T', 'Y', 'X')
        assert result.shape == (len(time_data), len(lat_data), len(lon_data))

        # Check coordinates are properly named
        assert 'T' in result.coords
        assert 'Y' in result.coords
        assert 'X' in result.coords

        # Check cf_xarray detection works
        assert result.cf['T'].name == 'T'
        assert result.cf['Y'].name == 'Y'
        assert result.cf['X'].name == 'X'

    def test_mesh_day_length_coordinate_values_preserved(self):
        """Test that coordinate values are preserved in the result."""
        time_data = pd.date_range('2023-06-01', periods=2, freq='ME')
        lat_data = np.array([0, 45])
        lon_data = np.array([0])

        time = new_time(time_data)
        latitude = new_latitude(lat_data)
        longitude = new_longitude(lon_data)

        result = _mesh_day_length(time, latitude, longitude, angle_horizon_sun=0)

        # Check coordinate values are preserved
        np.testing.assert_array_equal(result.coords['T'].values, time_data.values)
        np.testing.assert_array_equal(result.coords['Y'].values, lat_data)
        np.testing.assert_array_equal(result.coords['X'].values, lon_data)

    def test_mesh_day_length_attributes(self):
        """Test that result has proper attributes."""
        time = new_time(pd.date_range('2023-01-01', periods=1))
        latitude = new_latitude(np.array([0]))
        longitude = new_longitude(np.array([0]))

        result = _mesh_day_length(time, latitude, longitude, angle_horizon_sun=0)

        # Check attributes
        assert 'long_name' in result.attrs
        assert 'standard_name' in result.attrs
        assert 'units' in result.attrs
        assert result.attrs['standard_name'] == 'day_length'
        assert result.attrs['units'] == 'day'

    def test_mesh_day_length_physical_constraints(self):
        """Test that day length values are within physical constraints."""
        time_data = pd.date_range('2023-06-21', periods=1)  # Summer solstice
        lat_data = np.array([0, 45, -45])  # Equator and mid-latitudes
        lon_data = np.array([0])

        time = new_time(time_data)
        latitude = new_latitude(lat_data)
        longitude = new_longitude(lon_data)

        result = _mesh_day_length(time, latitude, longitude, angle_horizon_sun=0)

        # Day length should be between 0 and 1 day (values are in days, not hours)
        assert np.all(result.values >= 0)
        assert np.all(result.values <= 1)

        # At equator, day length should be close to 0.5 days (12 hours)
        equator_day_length = result.sel(Y=0, X=0).values[0]
        assert abs(equator_day_length - 0.5) < 0.05  # Within ~1.2 hours of 12

    def test_mesh_day_length_seasonal_variation(self):
        """Test that day length shows expected seasonal variation."""
        # Test at 45°N latitude through the year
        time_data = pd.date_range('2023-01-01', periods=12, freq='ME')
        lat_data = np.array([45])  # 45°N
        lon_data = np.array([0])

        time = new_time(time_data)
        latitude = new_latitude(lat_data)
        longitude = new_longitude(lon_data)

        result = _mesh_day_length(time, latitude, longitude, angle_horizon_sun=0)

        day_lengths = result.values[:, 0, 0]  # Extract time series

        # Should have seasonal variation
        min_day_length = np.min(day_lengths)
        max_day_length = np.max(day_lengths)

        # At 45°N, should have significant seasonal variation (values in days)
        assert max_day_length - min_day_length > 0.17  # More than ~4 hours difference

    def test_mesh_day_length_different_angles(self):
        """Test mesh_day_length with different horizon angles."""
        time = new_time(pd.date_range('2023-06-01', periods=1))
        latitude = new_latitude(np.array([45]))
        longitude = new_longitude(np.array([0]))

        # Test different horizon angles
        for angle in [0, -6, -12, -18]:  # Civil, nautical, astronomical twilight
            result = _mesh_day_length(time, latitude, longitude, angle_horizon_sun=angle)

            assert isinstance(result, xr.DataArray)
            assert result.dims == ('T', 'Y', 'X')
            assert np.all(result.values > 0)

    def test_mesh_day_length_no_old_coordinate_names(self):
        """Test that result doesn't contain old coordinate names."""
        time = new_time(pd.date_range('2023-01-01', periods=1))
        latitude = new_latitude(np.array([0]))
        longitude = new_longitude(np.array([0]))

        result = _mesh_day_length(time, latitude, longitude, angle_horizon_sun=0)

        # Should not contain old names
        assert 'time' not in result.coords
        assert 'latitude' not in result.coords
        assert 'longitude' not in result.coords
        assert 'time' not in result.dims
        assert 'latitude' not in result.dims
        assert 'longitude' not in result.dims

    def test_mesh_day_length_various_coordinate_sizes(self):
        """Test with various coordinate array sizes."""
        test_cases = [
            (1, 1, 1),  # Single point
            (2, 2, 2),  # Small 3D grid
            (12, 3, 2),  # Year of data, multiple locations
        ]

        for n_time, n_lat, n_lon in test_cases:
            time_data = pd.date_range('2023-01-01', periods=n_time, freq='ME')
            lat_data = np.linspace(-60, 60, n_lat)
            lon_data = np.linspace(-180, 180, n_lon)

            time = new_time(time_data)
            latitude = new_latitude(lat_data)
            longitude = new_longitude(lon_data)

            result = _mesh_day_length(time, latitude, longitude, angle_horizon_sun=0)

            assert result.shape == (n_time, n_lat, n_lon)
            assert result.dims == ('T', 'Y', 'X')

    @pytest.mark.parametrize("angle_horizon_sun", [0, -6, -12, -18])
    def test_mesh_day_length_parametrized_angles(self, angle_horizon_sun):
        """Test mesh_day_length with various horizon angles."""
        time = new_time(pd.date_range('2023-06-21', periods=1))
        latitude = new_latitude(np.array([0, 30, 60]))
        longitude = new_longitude(np.array([0]))

        result = _mesh_day_length(time, latitude, longitude, angle_horizon_sun=angle_horizon_sun)

        assert result.dims == ('T', 'Y', 'X')
        assert np.all(result.values >= 0)
        assert np.all(result.values <= 24)

    def test_mesh_day_length_integration_with_fixtures(self, sample_coordinates):
        """Test integration with conftest.py coordinate fixtures."""
        time = sample_coordinates['time']
        latitude = sample_coordinates['latitude']
        longitude = sample_coordinates['longitude']

        # Limit to smaller subset for faster test
        time_subset = time.isel(T=slice(0, 3))
        lat_subset = latitude.isel(Y=slice(0, 3))
        lon_subset = longitude.isel(X=slice(0, 3))

        result = _mesh_day_length(time_subset, lat_subset, lon_subset, angle_horizon_sun=0)

        assert isinstance(result, xr.DataArray)
        assert result.dims == ('T', 'Y', 'X')
        assert 'T' in result.coords
        assert 'Y' in result.coords
        assert 'X' in result.coords

    def test_mesh_day_length_with_different_time_frequencies(self):
        """Test day length calculation with different time frequencies."""
        latitude = new_latitude(np.array([45]))
        longitude = new_longitude(np.array([0]))

        # Test daily frequency
        daily_time = new_time(pd.date_range('2023-06-01', periods=7, freq='D'))
        daily_result = _mesh_day_length(daily_time, latitude, longitude, angle_horizon_sun=0)
        assert daily_result.shape[0] == 7

        # Test monthly frequency
        monthly_time = new_time(pd.date_range('2023-01-01', periods=12, freq='ME'))
        monthly_result = _mesh_day_length(monthly_time, latitude, longitude, angle_horizon_sun=0)
        assert monthly_result.shape[0] == 12
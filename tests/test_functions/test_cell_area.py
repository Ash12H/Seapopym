"""Tests for cell_area function with standardized coordinates."""

import pytest
import numpy as np
import xarray as xr

from seapopym.function.cell_area import _mesh_cell_area
from seapopym.standard.coordinates import new_latitude, new_longitude


@pytest.mark.scientific
class TestCellArea:
    """Test class for cell area calculation functions."""

    def test_mesh_cell_area_with_standardized_coordinates(self):
        """Test _mesh_cell_area with standardized Y/X coordinates."""
        # Create standardized coordinates
        lat_data = np.arange(-10, 11, 10)  # [-10, 0, 10]
        lon_data = np.arange(-20, 21, 20)  # [-20, 0, 20]

        latitude = new_latitude(lat_data)
        longitude = new_longitude(lon_data)

        # Test the function
        result = _mesh_cell_area(latitude, longitude, resolution=1.0)

        # Check result structure
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('Y', 'X')
        assert result.shape == (len(lat_data), len(lon_data))

        # Check coordinates are properly named
        assert 'Y' in result.coords
        assert 'X' in result.coords
        assert result.coords['Y'].name == 'Y'
        assert result.coords['X'].name == 'X'

        # Check cf_xarray detection works
        assert result.cf['Y'].name == 'Y'
        assert result.cf['X'].name == 'X'

    def test_mesh_cell_area_coordinate_values_preserved(self):
        """Test that coordinate values are preserved in the result."""
        lat_data = np.array([0, 10, 20])
        lon_data = np.array([-10, 0, 10])

        latitude = new_latitude(lat_data)
        longitude = new_longitude(lon_data)

        result = _mesh_cell_area(latitude, longitude, resolution=1.0)

        # Check coordinate values are preserved
        np.testing.assert_array_equal(result.coords['Y'].values, lat_data)
        np.testing.assert_array_equal(result.coords['X'].values, lon_data)

    def test_mesh_cell_area_attributes(self):
        """Test that result has proper attributes."""
        latitude = new_latitude(np.array([0, 10]))
        longitude = new_longitude(np.array([0, 10]))

        result = _mesh_cell_area(latitude, longitude, resolution=1.0)

        # Check attributes
        assert 'long_name' in result.attrs
        assert 'standard_name' in result.attrs
        assert 'units' in result.attrs
        assert result.attrs['standard_name'] == 'cell_area'

    def test_mesh_cell_area_data_calculation(self):
        """Test that cell area calculation produces reasonable values."""
        # Use equatorial latitude for predictable results
        lat_data = np.array([0])  # Equator
        lon_data = np.array([0, 1])  # 1 degree longitude difference

        latitude = new_latitude(lat_data)
        longitude = new_longitude(lon_data)

        result = _mesh_cell_area(latitude, longitude, resolution=1.0)

        # At equator, 1 degree longitude ≈ 111 km
        # So cell area should be positive and reasonable
        assert np.all(result.values > 0)
        assert np.all(np.isfinite(result.values))

    def test_mesh_cell_area_different_resolutions(self):
        """Test mesh_cell_area with different resolution values."""
        latitude = new_latitude(np.array([0, 10]))
        longitude = new_longitude(np.array([0, 10]))

        # Test different resolutions
        for resolution in [0.5, 1.0, 2.0]:
            result = _mesh_cell_area(latitude, longitude, resolution=resolution)

            assert isinstance(result, xr.DataArray)
            assert result.dims == ('Y', 'X')
            assert np.all(result.values > 0)

    def test_mesh_cell_area_various_coordinate_sizes(self):
        """Test with various coordinate array sizes."""
        test_cases = [
            (np.array([0]), np.array([0])),  # Single point
            (np.array([0, 10]), np.array([0, 10])),  # 2x2 grid
            (np.arange(-30, 31, 30), np.arange(-60, 61, 60)),  # 3x3 grid
        ]

        for lat_data, lon_data in test_cases:
            latitude = new_latitude(lat_data)
            longitude = new_longitude(lon_data)

            result = _mesh_cell_area(latitude, longitude, resolution=1.0)

            assert result.shape == (len(lat_data), len(lon_data))
            assert result.dims == ('Y', 'X')

    def test_mesh_cell_area_no_old_coordinate_names(self):
        """Test that result doesn't contain old coordinate names."""
        latitude = new_latitude(np.array([0, 10]))
        longitude = new_longitude(np.array([0, 10]))

        result = _mesh_cell_area(latitude, longitude, resolution=1.0)

        # Should not contain old names
        assert 'latitude' not in result.coords
        assert 'longitude' not in result.coords
        assert 'latitude' not in result.dims
        assert 'longitude' not in result.dims

    @pytest.mark.parametrize("lat_range,lon_range", [
        ((-10, 11, 10), (-10, 11, 10)),
        ((-90, 91, 30), (-180, 181, 60)),
        ((0, 1, 1), (0, 1, 1))
    ])
    def test_mesh_cell_area_parametrized(self, lat_range, lon_range):
        """Test mesh_cell_area with various coordinate ranges."""
        lat_data = np.arange(*lat_range)
        lon_data = np.arange(*lon_range)

        latitude = new_latitude(lat_data)
        longitude = new_longitude(lon_data)

        result = _mesh_cell_area(latitude, longitude, resolution=1.0)

        assert result.dims == ('Y', 'X')
        assert result.shape == (len(lat_data), len(lon_data))
        assert np.all(result.values > 0)  # All areas should be positive

    def test_mesh_cell_area_integration_with_fixtures(self, sample_coordinates):
        """Test integration with conftest.py coordinate fixtures."""
        latitude = sample_coordinates['latitude']
        longitude = sample_coordinates['longitude']

        result = _mesh_cell_area(latitude, longitude, resolution=1.0)

        assert isinstance(result, xr.DataArray)
        assert result.dims == ('Y', 'X')
        assert 'Y' in result.coords
        assert 'X' in result.coords
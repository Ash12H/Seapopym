"""Tests for ForcingUnit coordinate standardization."""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

from seapopym.configuration.no_transport.forcing_parameter import ForcingUnit


@pytest.mark.coordinates
class TestForcingUnit:
    """Test class for ForcingUnit coordinate standardization."""

    def test_standardize_coordinates_from_old_names(self, sample_dataarray_old_coords):
        """Test automatic coordinate standardization from old names."""
        # Create ForcingUnit with old coordinate names
        forcing_unit = ForcingUnit(forcing=sample_dataarray_old_coords)

        # Check that coordinates are standardized
        coords = list(forcing_unit.forcing.coords)
        assert 'T' in coords
        assert 'Y' in coords
        assert 'X' in coords

        # Check that old names are gone
        assert 'time' not in coords
        assert 'latitude' not in coords
        assert 'longitude' not in coords

    def test_standardize_coordinates_already_standardized(self, sample_dataarray_standardized):
        """Test that already standardized coordinates are left unchanged."""
        original_coords = list(sample_dataarray_standardized.coords)

        forcing_unit = ForcingUnit(forcing=sample_dataarray_standardized)

        # Coordinates should remain the same
        assert list(forcing_unit.forcing.coords) == original_coords
        assert 'T' in forcing_unit.forcing.coords
        assert 'Y' in forcing_unit.forcing.coords
        assert 'X' in forcing_unit.forcing.coords

    def test_cf_xarray_detection_works(self, sample_dataarray_old_coords):
        """Test that cf_xarray correctly detects standardized coordinates."""
        forcing_unit = ForcingUnit(forcing=sample_dataarray_old_coords)

        # Test cf_xarray detection
        assert forcing_unit.forcing.cf['T'].name == 'T'
        assert forcing_unit.forcing.cf['Y'].name == 'Y'
        assert forcing_unit.forcing.cf['X'].name == 'X'

    def test_attributes_preserved_during_standardization(self, sample_dataarray_old_coords):
        """Test that coordinate attributes are preserved during standardization."""
        forcing_unit = ForcingUnit(forcing=sample_dataarray_old_coords)

        # Check that attributes are preserved
        t_coord = forcing_unit.forcing.coords['T']
        y_coord = forcing_unit.forcing.coords['Y']
        x_coord = forcing_unit.forcing.coords['X']

        assert t_coord.attrs.get('standard_name') == 'time'
        assert y_coord.attrs.get('standard_name') == 'latitude'
        assert x_coord.attrs.get('standard_name') == 'longitude'
        assert y_coord.attrs.get('units') == 'degrees_north'
        assert x_coord.attrs.get('units') == 'degrees_east'

    def test_data_values_unchanged_during_standardization(self, sample_dataarray_old_coords):
        """Test that data values are unchanged during coordinate standardization."""
        original_data = sample_dataarray_old_coords.values.copy()
        original_time_values = sample_dataarray_old_coords.coords['time'].values.copy()
        original_lat_values = sample_dataarray_old_coords.coords['latitude'].values.copy()
        original_lon_values = sample_dataarray_old_coords.coords['longitude'].values.copy()

        forcing_unit = ForcingUnit(forcing=sample_dataarray_old_coords)

        # Data should be unchanged
        np.testing.assert_array_equal(forcing_unit.forcing.values, original_data)
        np.testing.assert_array_equal(forcing_unit.forcing.coords['T'].values, original_time_values)
        np.testing.assert_array_equal(forcing_unit.forcing.coords['Y'].values, original_lat_values)
        np.testing.assert_array_equal(forcing_unit.forcing.coords['X'].values, original_lon_values)

    def test_standardize_coordinates_with_layer_dimension(self):
        """Test standardization with Z (layer) dimension."""
        # Create DataArray with layer dimension
        time_data = pd.date_range('2023-01-01', periods=3, freq='ME')
        lat_data = np.arange(-10, 11, 10)
        lon_data = np.arange(-10, 11, 10)
        layer_data = [0, -50, -100]

        data = np.random.random((len(time_data), len(lat_data), len(lon_data), len(layer_data)))

        da = xr.DataArray(
            data,
            coords={
                'time': ('time', time_data, {'standard_name': 'time', 'axis': 'T'}),
                'latitude': ('latitude', lat_data, {'standard_name': 'latitude', 'axis': 'Y'}),
                'longitude': ('longitude', lon_data, {'standard_name': 'longitude', 'axis': 'X'}),
                'layer': ('layer', layer_data, {'standard_name': 'depth', 'axis': 'Z'})
            },
            dims=['time', 'latitude', 'longitude', 'layer']
        )

        forcing_unit = ForcingUnit(forcing=da)

        # Check all coordinates are standardized
        coords = list(forcing_unit.forcing.coords)
        assert 'T' in coords
        assert 'Y' in coords
        assert 'X' in coords
        assert 'Z' in coords

        # Check old names are gone
        assert 'time' not in coords
        assert 'latitude' not in coords
        assert 'longitude' not in coords
        assert 'layer' not in coords

    def test_standardize_coordinates_partial_standardization(self):
        """Test with mixed old/new coordinate names."""
        time_data = pd.date_range('2023-01-01', periods=3, freq='ME')
        lat_data = np.arange(-10, 11, 10)
        lon_data = np.arange(-10, 11, 10)

        data = np.random.random((len(time_data), len(lat_data), len(lon_data)))

        # Mix of old and new names
        da = xr.DataArray(
            data,
            coords={
                'T': ('T', time_data, {'standard_name': 'time', 'axis': 'T'}),  # Already standardized
                'latitude': ('latitude', lat_data, {'standard_name': 'latitude', 'axis': 'Y'}),  # Old name
                'X': ('X', lon_data, {'standard_name': 'longitude', 'axis': 'X'})  # Already standardized
            },
            dims=['T', 'latitude', 'X']
        )

        forcing_unit = ForcingUnit(forcing=da)

        # All should be standardized
        coords = list(forcing_unit.forcing.coords)
        assert coords == ['T', 'Y', 'X']

    def test_no_cf_attributes_fallback(self):
        """Test fallback when cf_xarray can't detect coordinates."""
        # DataArray without proper CF attributes
        data = np.random.random((3, 5, 7))
        da = xr.DataArray(
            data,
            coords={
                'dim_0': range(3),
                'dim_1': range(5),
                'dim_2': range(7)
            },
            dims=['dim_0', 'dim_1', 'dim_2']
        )

        # Should not crash, should return original DataArray
        forcing_unit = ForcingUnit(forcing=da)

        # Should keep original coordinates when cf_xarray can't detect
        original_coords = ['dim_0', 'dim_1', 'dim_2']
        assert list(forcing_unit.forcing.coords) == original_coords

    def test_fixture_integration(self, sample_forcing_unit_old_coords):
        """Test integration with conftest.py fixtures."""
        # Test with fixture that has old coordinate names
        coords = list(sample_forcing_unit_old_coords.forcing.coords)

        assert 'T' in coords
        assert 'Y' in coords
        assert 'X' in coords
        assert 'time' not in coords
        assert 'latitude' not in coords
        assert 'longitude' not in coords
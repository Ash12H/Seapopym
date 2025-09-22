"""Tests for CoordinateAuthority integration with template system.

Tests validate that templates generated with CoordinateAuthority integration
preserve coordinate attributes correctly.
"""

import numpy as np
import pytest
import xarray as xr

from seapopym.core.template import Template, TemplateUnit
from seapopym.standard.attributs import biomass_desc
from seapopym.standard.coordinates import new_latitude, new_longitude, new_time
from seapopym.standard.labels import CoordinatesLabels, ForcingLabels


class TestTemplateIntegration:
    """Test CoordinateAuthority integration with template system."""

    def create_test_state(self) -> xr.Dataset:
        """Create a test SeapopymState with proper coordinates."""
        time_coord = new_time([1, 2, 3])
        lat_coord = new_latitude([10.0, 20.0])
        lon_coord = new_longitude([100.0, 110.0, 120.0])

        # Create some test data
        temp_data = np.random.random((3, 2, 3))

        return xr.Dataset(
            {
                "temperature": (["T", "Y", "X"], temp_data)
            },
            coords={"T": time_coord, "Y": lat_coord, "X": lon_coord}
        )

    def test_template_unit_preserves_coordinate_attrs(self):
        """Test that TemplateUnit preserves coordinate attributes after generation."""
        state = self.create_test_state()

        # Create a template unit
        template_unit = TemplateUnit(
            name=ForcingLabels.biomass,
            attrs=biomass_desc,
            dims=[CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X]
        )

        # Generate template
        generated = template_unit.generate(state)

        # Check that coordinate attributes are preserved
        # Note: After fix, the template should preserve coordinate attributes
        assert generated.coords["T"].attrs["axis"] == "T"
        assert generated.coords["Y"].attrs["axis"] == "Y"
        assert generated.coords["X"].attrs["axis"] == "X"
        assert generated.coords["Y"].attrs["units"] == "degrees_north"
        assert generated.coords["X"].attrs["units"] == "degrees_east"

    def test_template_preserves_coordinate_attrs(self):
        """Test that Template class preserves coordinate attributes."""
        state = self.create_test_state()

        # Create template units
        biomass_template = TemplateUnit(
            name=ForcingLabels.biomass,
            attrs=biomass_desc,
            dims=[CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X]
        )

        # Create template with multiple units
        template = Template(template_unit=[biomass_template])

        # Generate template dataset
        generated = template.generate(state)

        # Check that coordinate attributes are preserved in the dataset
        assert generated.coords["T"].attrs["axis"] == "T"
        assert generated.coords["Y"].attrs["axis"] == "Y"
        assert generated.coords["X"].attrs["axis"] == "X"
        assert generated.coords["Y"].attrs["units"] == "degrees_north"
        assert generated.coords["X"].attrs["units"] == "degrees_east"

    def test_template_generation_with_missing_coordinate_attrs(self):
        """Test template generation when input state has missing coordinate attributes."""
        # Create state with coordinates missing some attributes
        time_data = np.array([1, 2, 3])
        lat_data = np.array([10.0, 20.0])
        lon_data = np.array([100.0, 110.0, 120.0])

        # Create coordinates with missing attributes
        time_coord = xr.DataArray(time_data, dims=["T"], name="T")  # missing attributes
        lat_coord = xr.DataArray(lat_data, dims=["Y"], name="Y")   # missing attributes
        lon_coord = xr.DataArray(lon_data, dims=["X"], name="X")   # missing attributes

        # Create dataset
        state = xr.Dataset(
            {
                "temperature": (["T", "Y", "X"], np.random.random((3, 2, 3)))
            },
            coords={"T": time_coord, "Y": lat_coord, "X": lon_coord}
        )

        # Create template
        template_unit = TemplateUnit(
            name=ForcingLabels.biomass,
            attrs=biomass_desc,
            dims=[CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X]
        )

        # Generate template - should restore missing attributes
        generated = template_unit.generate(state)

        # Check that missing attributes were restored
        assert generated.coords["T"].attrs["axis"] == "T"
        assert generated.coords["Y"].attrs["axis"] == "Y"
        assert generated.coords["X"].attrs["axis"] == "X"

    def test_template_coordinate_validation_idempotent(self):
        """Test that coordinate validation is idempotent (multiple calls don't change result)."""
        state = self.create_test_state()

        template_unit = TemplateUnit(
            name=ForcingLabels.biomass,
            attrs=biomass_desc,
            dims=[CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X]
        )

        # Generate template twice
        generated1 = template_unit.generate(state)
        generated2 = template_unit.generate(state)

        # Results should be identical
        xr.testing.assert_identical(generated1, generated2)
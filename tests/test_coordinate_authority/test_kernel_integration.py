"""Tests for CoordinateAuthority integration with kernel system.

Tests validate that the asymmetry between sequential and parallel execution
is resolved and that coordinate attributes are preserved in both modes.
"""

import numpy as np
import pytest
import xarray as xr

from seapopym.core.kernel import Kernel, KernelUnit
from seapopym.core.template import Template, TemplateUnit
from seapopym.standard.attributs import biomass_desc
from seapopym.standard.coordinates import new_latitude, new_longitude, new_time
from seapopym.standard.labels import CoordinatesLabels, ForcingLabels


class TestKernelIntegration:
    """Test CoordinateAuthority integration with kernel execution."""

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

    def create_test_function(self):
        """Create a simple test function for kernel testing."""
        def test_func(state: xr.Dataset) -> xr.Dataset:
            # Simple function that creates biomass from temperature
            biomass_data = state["temperature"] * 0.1
            return xr.Dataset({ForcingLabels.biomass: biomass_data})

        return test_func

    def test_kernel_unit_sequential_preserves_coordinates(self):
        """Test that KernelUnit in sequential mode preserves coordinate attributes."""
        state = self.create_test_state()
        test_func = self.create_test_function()

        # Create template
        template_unit = TemplateUnit(
            name=ForcingLabels.biomass,
            attrs=biomass_desc,
            dims=[CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X]
        )

        # Create kernel unit for sequential execution
        kernel_unit = KernelUnit(
            name="test_kernel",
            template=Template(template_unit=[template_unit]),
            function=test_func,
            parallel=False  # Sequential mode
        )

        # Run kernel unit
        result = kernel_unit.run(state)

        # Check that coordinate attributes are preserved
        assert result.coords["T"].attrs["axis"] == "T"
        assert result.coords["Y"].attrs["axis"] == "Y"
        assert result.coords["X"].attrs["axis"] == "X"
        assert result.coords["Y"].attrs["units"] == "degrees_north"
        assert result.coords["X"].attrs["units"] == "degrees_east"

    def test_kernel_unit_parallel_preserves_coordinates(self):
        """Test that KernelUnit in parallel mode preserves coordinate attributes."""
        state = self.create_test_state()
        test_func = self.create_test_function()

        # Create template
        template_unit = TemplateUnit(
            name=ForcingLabels.biomass,
            attrs=biomass_desc,
            dims=[CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X]
        )

        # Create kernel unit for parallel execution
        kernel_unit = KernelUnit(
            name="test_kernel",
            template=Template(template_unit=[template_unit]),
            function=test_func,
            parallel=True  # Parallel mode
        )

        # Run kernel unit
        result = kernel_unit.run(state)

        # Check that coordinate attributes are preserved
        assert result.coords["T"].attrs["axis"] == "T"
        assert result.coords["Y"].attrs["axis"] == "Y"
        assert result.coords["X"].attrs["axis"] == "X"
        assert result.coords["Y"].attrs["units"] == "degrees_north"
        assert result.coords["X"].attrs["units"] == "degrees_east"

    def test_sequential_parallel_consistency(self):
        """Test that sequential and parallel modes produce consistent coordinate attributes.

        This is the key test that validates the fix for the asymmetry issue.
        """
        state = self.create_test_state()
        test_func = self.create_test_function()

        # Create template
        template_unit = TemplateUnit(
            name=ForcingLabels.biomass,
            attrs=biomass_desc,
            dims=[CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X]
        )

        # Create kernel units for both modes
        kernel_sequential = KernelUnit(
            name="test_kernel_seq",
            template=Template(template_unit=[template_unit]),
            function=test_func,
            parallel=False
        )

        kernel_parallel = KernelUnit(
            name="test_kernel_par",
            template=Template(template_unit=[template_unit]),
            function=test_func,
            parallel=True
        )

        # Run both modes
        result_sequential = kernel_sequential.run(state)
        result_parallel = kernel_parallel.run(state)

        # Check that coordinate attributes are identical between modes
        for coord_name in ["T", "Y", "X"]:
            seq_attrs = result_sequential.coords[coord_name].attrs
            par_attrs = result_parallel.coords[coord_name].attrs

            # All coordinate attributes should be identical
            assert seq_attrs == par_attrs, f"Coordinate {coord_name} attributes differ between sequential and parallel modes"

    def test_kernel_preserves_coordinates_after_merge(self):
        """Test that Kernel preserves coordinate attributes after state merging."""
        state = self.create_test_state()

        def create_biomass(state: xr.Dataset) -> xr.Dataset:
            biomass_data = state["temperature"] * 0.1
            return xr.Dataset({ForcingLabels.biomass: biomass_data})

        # Create kernel unit
        template_unit = TemplateUnit(
            name=ForcingLabels.biomass,
            attrs=biomass_desc,
            dims=[CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X]
        )

        kernel_unit_class = type("TestKernelUnit", (KernelUnit,), {
            "__init__": lambda self, chunk, parallel=False: KernelUnit.__init__(
                self,
                name="biomass_kernel",
                template=Template(template_unit=[template_unit]),
                function=create_biomass,
                parallel=parallel
            )
        })

        # Create kernel
        kernel = Kernel(
            kernel_unit=[kernel_unit_class],
            chunk={},
            parallel=False
        )

        # Run kernel
        result_state = kernel.run(state)

        # Check that original coordinates are preserved
        assert result_state.coords["T"].attrs["axis"] == "T"
        assert result_state.coords["Y"].attrs["axis"] == "Y"
        assert result_state.coords["X"].attrs["axis"] == "X"

        # Check that the result contains both original and new variables
        assert "temperature" in result_state
        assert ForcingLabels.biomass in result_state

    def test_coordinate_integrity_after_variable_removal(self):
        """Test that coordinate integrity is maintained after removing variables from state."""
        state = self.create_test_state()

        def create_biomass(state: xr.Dataset) -> xr.Dataset:
            biomass_data = state["temperature"] * 0.1
            return xr.Dataset({ForcingLabels.biomass: biomass_data})

        # Create kernel unit that removes temperature after creating biomass
        template_unit = TemplateUnit(
            name=ForcingLabels.biomass,
            attrs=biomass_desc,
            dims=[CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X]
        )

        kernel_unit_class = type("TestKernelUnit", (KernelUnit,), {
            "__init__": lambda self, chunk, parallel=False: KernelUnit.__init__(
                self,
                name="biomass_kernel",
                template=Template(template_unit=[template_unit]),
                function=create_biomass,
                to_remove_from_state=["temperature"],  # Remove temperature after processing
                parallel=parallel
            )
        })

        # Create kernel
        kernel = Kernel(
            kernel_unit=[kernel_unit_class],
            chunk={},
            parallel=False
        )

        # Run kernel
        result_state = kernel.run(state)

        # Check that coordinates are still intact after variable removal
        assert result_state.coords["T"].attrs["axis"] == "T"
        assert result_state.coords["Y"].attrs["axis"] == "Y"
        assert result_state.coords["X"].attrs["axis"] == "X"

        # Check that temperature was removed but biomass remains
        assert "temperature" not in result_state
        assert ForcingLabels.biomass in result_state
"""The no transport model with acidity-induced mortality."""

from __future__ import annotations
<<<<<<< HEAD
from seapopym.model.no_transport_model import NoTransportModel
from seapopym.configuration.acidity.acidity_configuration import AcidityConfiguration
from seapopym.function import generator
from seapopym.function.generator.mask import apply_mask_to_state
from seapopym.standard.coordinates import reorder_dims
from seapopym.function.core.kernel import Kernel

class AcidityModel(NoTransportModel):
    """A pteropod 1D model that takes into account the mortality due to ocean acidification"""

    def __init__(self: AcidityModel,configuration: AcidityConfiguration):
        self._configuration = configuration
        self.state = apply_mask_to_state(reorder_dims(configuration.model_parameters))

        chunk = self.configuration.environment_parameters.chunk.as_dict()
        # ordre important
        self._kernel = Kernel(
            [
                generator.global_mask_kernel(chunk=chunk),
                generator.mask_by_fgroup_kernel(chunk=chunk),
                generator.day_length_kernel(
                    chunk=chunk, angle_horizon_sun=configuration.kernel_parameters.angle_horizon_sun
                ),
                generator.average_temperature_kernel(chunk=chunk),
                generator.average_acidity_kernel(chunk=chunk),
                generator.apply_coefficient_to_primary_production_kernel(chunk=chunk),
                generator.min_temperature_kernel(chunk=chunk),
                generator.mask_temperature_kernel(chunk=chunk),
                generator.cell_area_kernel(chunk=chunk),
                generator.mortality_acidity_field_kernel(chunk=chunk),
                generator.production_kernel(
                    chunk=chunk,
                    export_preproduction=configuration.kernel_parameters.compute_preproduction,
                    export_initial_production=configuration.kernel_parameters.compute_initial_conditions,
                ),
                generator.biomass_kernel(chunk=chunk),
            ]
        )
=======

from typing import TYPE_CHECKING

from seapopym import function
from seapopym.core.kernel import kernel_factory
from seapopym.function.apply_mask_to_state import apply_mask_to_state
from seapopym.model.no_transport_model import NoTransportModel
from seapopym.standard.coordinates import reorder_dims

if TYPE_CHECKING:
    from seapopym.configuration.acidity import AcidityConfiguration

AcidityKernel = kernel_factory(
    class_name="AcidityKernel",
    kernel_unit=[
        function.GlobalMaskKernel,
        function.MaskByFunctionalGroupKernel,
        function.DayLengthKernel,
        function.AverageTemperatureKernel,
        function.AverageAcidityKernel,
        function.PrimaryProductionByFgroupKernel,
        function.MinTemperatureByCohortKernel,
        function.MaskTemperatureKernel,
        function.MortalityTemperatureAcidityKernel,
        function.production.ProductionKernel,
        function.BiomassKernel,
    ],
)


class AcidityModel(NoTransportModel):
    """A pteropod 1D model that takes into account the mortality due to ocean acidification."""

    @classmethod
    def from_configuration(cls: type[AcidityModel], configuration: AcidityConfiguration) -> AcidityModel:
        """Create a model from a configuration."""
        return cls(
            environment=configuration.environment,
            state=apply_mask_to_state(reorder_dims(configuration.state)),
            kernel=AcidityKernel(chunk=configuration.environment.chunk.as_dict()),
        )
>>>>>>> origin/master

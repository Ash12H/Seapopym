"""The no transport model with acidity-induced mortality."""

from __future__ import annotations
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
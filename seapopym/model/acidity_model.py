"""The no transport model with acidity-induced mortality."""

from __future__ import annotations

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
        state = configuration.state
        return cls(state=state, kernel=AcidityKernel(chunk=state.chunksizes))

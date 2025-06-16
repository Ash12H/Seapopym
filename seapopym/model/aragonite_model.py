"""The no transport model with acidity-induced mortality.
In this version, the acidity induced mortality is targeting the junevile population.
To do that, the primary production is multiplied with a function of the aragonite.
(note that aragonitite in this model refers to Omega Aragonite - Aragonite sursaturation)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from seapopym import function
from seapopym.core.kernel import kernel_factory
from seapopym.function.apply_mask_to_state import apply_mask_to_state
from seapopym.model.no_transport_model import NoTransportModel
from seapopym.standard.coordinates import reorder_dims

if TYPE_CHECKING:
    from seapopym.configuration.aragonite import AragoniteConfiguration


AragoniteKernel = kernel_factory(
    class_name="AragoniteKernel",
    kernel_unit=[
        function.GlobalMaskKernel,
        function.MaskByFunctionalGroupKernel,
        function.DayLengthKernel,
        function.AverageTemperatureKernel,
        function.AverageAcidityKernel,
        function.PrimaryProductionByFgroupKernelAragonite,
        function.MinTemperatureByCohortKernel,
        function.MaskTemperatureKernel,
        function.MortalityFieldKernel,
        function.production.ProductionKernel,
        function.BiomassKernel,
    ],
)


class AragoniteModel(NoTransportModel):
    """A pteropod 1D model that takes into account the junevile mortality due to ocean acidification."""

    @classmethod
    def from_configuration(cls: type[AragoniteModel], configuration: AragoniteConfiguration) -> AragoniteModel:
        """Create a model from a configuration."""
        return cls(
            environment=configuration.environment,
            state=apply_mask_to_state(reorder_dims(configuration.state)),
            kernel=AragoniteKernel(chunk=configuration.environment.chunk.as_dict()),
        )
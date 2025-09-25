"""The no transport model with acidity-induced mortality."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seapopym import function
from seapopym.core.kernel import kernel_factory
from seapopym.model.no_transport_model import NoTransportModel

if TYPE_CHECKING:
    from seapopym.configuration.acidity import AcidityConfiguration
    from seapopym.configuration.acidity_bed import AcidityBedConfiguration

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
        function.ProductionKernel,
        function.BiomassKernel,
    ],
)


class AcidityModel(NoTransportModel):
    """A pteropod 1D model that takes into account the mortality due to ocean acidification."""

    @classmethod
    def from_configuration(cls: type[AcidityModel], configuration: AcidityConfiguration) -> AcidityModel:
        """Create a model from a configuration."""
        state = configuration.state
        chunk = configuration.forcing.chunk.as_dict()
        parallel = configuration.forcing.parallel
        return cls(state=state, kernel=AcidityKernel(chunk=chunk, parallel=parallel))


AcidityBedKernel = kernel_factory(
    class_name="AcidityBedKernel",
    kernel_unit=[
        function.GlobalMaskKernel,
        function.MaskByFunctionalGroupKernel,
        function.DayLengthKernel,
        function.AverageTemperatureKernel,
        function.AverageAcidityKernel,
        function.PrimaryProductionByFgroupKernel,
        function.MinTemperatureByCohortKernel,
        function.MaskTemperatureKernel,
        function.SurvivalRateBednarsekKernel,
        function.MortalityTemperatureAcidityBedKernel,
        function.ProductionKernel,
        function.ApplySurvivalRateToRecruitmentKernel,
        function.BiomassKernel,
    ],
)


class AcidityBedModel(NoTransportModel):
    """A pteropod 1D model using Bednarsek et al. (2022) mortality equation for ocean acidification effects."""

    @classmethod
    def from_configuration(cls: type[AcidityBedModel], configuration: AcidityBedConfiguration) -> AcidityBedModel:
        """Create a model from a configuration."""
        state = configuration.state
        chunk = configuration.forcing.chunk.as_dict()
        parallel = configuration.forcing.parallel
        return cls(state=state, kernel=AcidityBedKernel(chunk=chunk, parallel=parallel))

"""The LMTL model without ADRE equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from seapopym import function
from seapopym.core.kernel import kernel_factory
from seapopym.model.base_model import BaseModel
from seapopym.standard.labels import ConfigurationLabels, ForcingLabels

if TYPE_CHECKING:
    import xarray as xr

    from seapopym.configuration.no_transport.configuration import NoTransportConfiguration
    from seapopym.standard.types import SeapopymState

pre_kernel = [
    function.GlobalMaskKernel,
    function.MaskByFunctionalGroupKernel,
    function.DayLengthKernel,
    function.AverageTemperatureKernel,
    function.PrimaryProductionByFgroupKernel,
    function.MinTemperatureByCohortKernel,
    function.MaskTemperatureKernel,
    function.MortalityFieldKernel,
]

NoTransportKernel = kernel_factory(
    class_name="NoTransportKernel",
    kernel_unit=[
        *pre_kernel,
        function.production.ProductionKernel,
        function.BiomassKernel,
    ],
)

NoTransportInitialConditionKernel = kernel_factory(
    class_name="NoTransportInitialConditionKernel",
    kernel_unit=[
        *pre_kernel,
        function.production.ProductionInitialConditionKernel,
        function.BiomassKernel,
    ],
)

NoTransportUnrecruitedKernel = kernel_factory(
    class_name="NoTransportUnrecruitedKernel",
    kernel_unit=[
        *pre_kernel,
        function.production.ProductionUnrecruitedKernel,
        function.BiomassKernel,
    ],
)


@dataclass
class NoTransportModel(BaseModel):
    """Implement the LMTL model without the transport (Advection-Diffusion)."""

    @classmethod
    def from_configuration(cls: type[NoTransportModel], configuration: NoTransportConfiguration) -> NoTransportModel:
        """Create a model from a configuration."""
        if configuration.kernel.compute_initial_conditions:
            kernel_class = NoTransportInitialConditionKernel
        elif configuration.kernel.compute_preproduction:
            kernel_class = NoTransportUnrecruitedKernel
        else:
            kernel_class = NoTransportKernel

        state = configuration.state
        chunk = state.chunksizes

        return cls(state=state, kernel=kernel_class(chunk=chunk))

    @property
    def template(self: NoTransportModel) -> SeapopymState:
        """The template getter."""
        return self.kernel.template(self.state)

    @property
    def expected_memory_usage(self: NoTransportModel) -> int:
        """The expected memory usage getter."""
        return f"The expected memory usage is {self.template.nbytes / 1e6:.2f} MB."

    def run(self: NoTransportModel) -> None:
        """Run the model. Wrapper of the pre-production, production and post-production processes."""
        self.state = self.kernel.run(self.state).persist()

    def export_initial_conditions(self: NoTransportModel) -> xr.Dataset:
        """Export the initial conditions."""
        if (
            not self.state[ConfigurationLabels.compute_initial_conditions]
            and not self.state[ConfigurationLabels.compute_preproduction]
        ):
            msg = (
                "To export initial conditions, the model must be run with the compute_initial_conditions or "
                "compute_preproduction flag set to True."
            )
            raise ValueError(msg)
        return self.state[[ForcingLabels.biomass, ForcingLabels.preproduction]].cf.isel(T=-1)


pre_kernel_light = [
    function.GlobalMaskKernel,
    function.mask_by_functional_group.MaskByFunctionalGroupKernelLight,
    function.DayLengthKernel,
    function.average_temperature.AverageTemperatureKernelLight,
    function.apply_coefficient_to_primary_production.PrimaryProductionByFgroupKernelLight,
    function.MinTemperatureByCohortKernel,
    function.mask_temperature.MaskTemperatureKernelLight,
    function.mortality_field.MortalityFieldKernelLight,
]

NoTransportKernelLight = kernel_factory(
    class_name="NoTransportLightKernel",
    kernel_unit=[
        *pre_kernel_light,
        function.production.ProductionKernelLight,
        function.biomass.BiomassKernelLight,
    ],
)


NoTransportInitialConditionKernelLight = kernel_factory(
    class_name="NoTransportInitialConditionKernel",
    kernel_unit=[
        *pre_kernel_light,
        function.production.ProductionInitialConditionKernelLight,
        function.biomass.BiomassKernelLight,
    ],
)

NoTransportUnrecruitedKernelLight = kernel_factory(
    class_name="NoTransportUnrecruitedKernel",
    kernel_unit=[
        *pre_kernel_light,
        function.production.ProductionUnrecruitedKernelLight,
        function.biomass.BiomassKernelLight,
    ],
)


@dataclass
class NoTransportLightModel(NoTransportModel):
    """Implement the LMTL model without the transport (Advection-Diffusion) and with light kernel."""

    @classmethod
    def from_configuration(
        cls: type[NoTransportLightModel], configuration: NoTransportConfiguration
    ) -> NoTransportLightModel:
        """Create a model from a configuration."""
        if configuration.kernel.compute_initial_conditions:
            kernel_class = NoTransportInitialConditionKernelLight
        elif configuration.kernel.compute_preproduction:
            kernel_class = NoTransportUnrecruitedKernelLight
        else:
            kernel_class = NoTransportKernelLight

        state = configuration.state
        chunk = state.chunksizes

        return cls(state=state, kernel=kernel_class(chunk=chunk))

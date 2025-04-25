"""The LMTL model without ADRE equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from seapopym import function
from seapopym.core.kernel import kernel_factory
from seapopym.function.apply_mask_to_state import apply_mask_to_state
from seapopym.logging.custom_logger import logger
from seapopym.model.base_model import BaseModel
from seapopym.standard.coordinates import reorder_dims

if TYPE_CHECKING:
    from seapopym.configuration.no_transport.configuration import NoTransportConfiguration
    from seapopym.configuration.no_transport.environment_parameter import EnvironmentParameter
    from seapopym.standard.types import SeapopymState


NoTransportKernel = kernel_factory(
    class_name="NoTransportKernel",
    kernel_unit=[
        function.GlobalMaskKernel,
        function.MaskByFunctionalGroupKernel,
        function.DayLengthKernel,
        function.AverageTemperatureKernel,
        function.PrimaryProductionByFgroupKernel,
        function.MinTemperatureByCohortKernel,
        function.MaskTemperatureKernel,
        function.MortalityFieldKernel,
        function.ProductionKernel,
        function.BiomassKernel,
    ],
)


@dataclass
class NoTransportModel(BaseModel):
    """Implement the LMTL model without the transport (Advection-Diffusion)."""

    environment: EnvironmentParameter

    @classmethod
    def from_configuration(cls: type[NoTransportModel], configuration: NoTransportConfiguration) -> NoTransportModel:
        """Create a model from a configuration."""
        return cls(
            environment=configuration.environment,
            state=apply_mask_to_state(reorder_dims(configuration.state)),
            kernel=NoTransportKernel(chunk=configuration.environment.chunk.as_dict()),
        )

    @property
    def template(self: NoTransportModel) -> SeapopymState:
        """The template getter."""
        return self.kernel.template(self.state)

    @property
    def expected_memory_usage(self: NoTransportModel) -> int:
        """The expected memory usage getter."""
        return f"The expected memory usage is {self.template.nbytes / 1e6:.2f} MB."

    def initialize_dask(self: NoTransportModel) -> None:
        """Initialize the client and configure the model to run in distributed mode."""
        logger.info("Initializing the client.")
        self.environment.client.initialize_client()
        self.state = self.state.chunk(self.environment.chunk.as_dict())

    def run(self: NoTransportModel) -> None:
        """Run the model. Wrapper of the pre-production, production and post-production processes."""
        self.state = self.kernel.run(self.state)

    def close(self: NoTransportModel) -> None:
        """Clean up the system. For example, it can be used to close dask.Client."""
        self.environment.client.close_client()

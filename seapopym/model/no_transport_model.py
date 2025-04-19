"""The LMTL model without ADRE equations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seapopym.function import generator
from seapopym.function.core.kernel import Kernel, kernel_factory
from seapopym.function.generator.apply_mask_to_state import apply_mask_to_state
from seapopym.logging.custom_logger import logger
from seapopym.model.base_model import BaseModel
from seapopym.plotter import base_functions as pfunctions
from seapopym.standard.coordinates import reorder_dims
from seapopym.writer import base_functions as wfunctions

if TYPE_CHECKING:
    from dask.distributed import Client

    from seapopym.configuration.no_transport.configuration import NoTransportConfiguration
    from seapopym.standard.types import SeapopymState


NoTransportKernel = kernel_factory(
    class_name="NoTransportKernel",
    kernel_unit=[
        generator.GlobalMaskKernel,
        generator.MaskByFunctionalGroupKernel,
        generator.DayLengthKernel,
        generator.AverageTemperatureKernel,
        generator.PrimaryProductionByFgroupKernel,
        generator.MinTemperatureByCohortKernel,
        generator.MaskTemperatureKernel,
        generator.CellAreaKernel,
        generator.MortalityFieldKernel,
        generator.ProductionKernel,
        generator.BiomassKernel,
    ],
)


class NoTransportModel(BaseModel):
    """Implement the LMTL model without the transport (Advection-Diffusion)."""

    def __init__(self: NoTransportModel, configuration: NoTransportConfiguration) -> None:
        """The constructor of the model allows the user to overcome the default parameters and client behaviors."""
        self._configuration = configuration
        self.state = apply_mask_to_state(reorder_dims(configuration.state))

        chunk = self.configuration.environment.chunk.as_dict()
        self._kernel = NoTransportKernel(chunk=chunk)

    @property
    def configuration(self: NoTransportModel) -> NoTransportConfiguration:
        """The configuration getter."""
        return self._configuration

    @property
    def client(self: NoTransportModel) -> Client | None:
        """The dask Client getter."""
        return self._configuration.environment.client.client

    @property
    def kernel(self: NoTransportModel) -> Kernel:
        """The kernel getter."""
        return self._kernel

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
        self.configuration.environment.client.initialize_client()
        chunk = self.configuration.environment.chunk.as_dict()
        self.state = self.state.chunk(chunk)
        logger.info("Scattering the data to the workers.")
        self.client.scatter(self.state)

    def run(self: NoTransportModel) -> None:
        """Run the model. Wrapper of the pre-production, production and post-production processes."""
        self.state = self.kernel.run(self.state)
        if self.client is not None:
            self.client.persist(self.state)

    def close(self: NoTransportModel) -> None:
        """Clean up the system. For example, it can be used to close dask.Client."""
        self.configuration.environment.client.close_client()

    # --- Export functions --- #

    export_state = wfunctions.export_state
    export_biomass = wfunctions.export_biomass
    export_initial_conditions = wfunctions.export_initial_conditions

    # --- Plot functions --- #

    plot_biomass = pfunctions.plot_biomass

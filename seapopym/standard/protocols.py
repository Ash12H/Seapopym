"""Protocol definitions for Seapopym architecture.

This module defines Protocol interfaces that replace Abstract Base Classes,
providing duck typing capabilities while maintaining type safety and interface contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import IO

    import xarray as xr


class ChunkParameterProtocol(Protocol):
    """Protocol for chunk parameter classes.

    Defines the interface for classes that manage chunking dimensions
    for parallel computation in xarray operations.
    """

    def as_dict(self) -> dict:
        """Format chunk parameters to a dictionary as expected by xarray.

        Returns:
            dict: Dictionary with dimension names as keys and chunk sizes as values.
        """
        ...


class KernelParameterProtocol(Protocol):
    """Protocol for kernel parameter classes.

    Defines the interface for classes that contain meta-parameters
    used to modify kernel function behavior. These parameters are
    integrated into the model state.

    Note: This protocol currently has no required methods as kernel
    parameters are primarily data containers.
    """

    pass


class ForcingUnitProtocol(Protocol):
    """Protocol for single forcing unit classes.

    Defines the interface for classes that represent a single forcing field
    (e.g., temperature, oxygen, primary production).
    """

    forcing: Any  # The forcing field data


class FunctionalGroupUnitProtocol(Protocol):
    """Protocol for single functional group unit classes.

    Defines the interface for classes that represent a single functional group
    with its migratory and functional type parameters.
    """

    name: str  # The name of the functional group
    migratory_type: Any  # Vertical migratory behavior parameters
    functional_type: Any  # Environment relationship parameters

    def to_dataset(self, timestep: int) -> xr.Dataset:
        """Convert functional group parameters to xarray Dataset.

        Args:
            timestep: Time step value for the conversion.

        Returns:
            xr.Dataset: Dataset containing the functional group parameters
                       for integration into SeapopymState.
        """
        ...


# Phase 2 Protocols - Level 2 (depend on Phase 1 protocols)

class ForcingParameterProtocol(Protocol):
    """Protocol for forcing parameter classes.

    Defines the interface for classes that manage forcing parameters,
    including parallel computation settings and chunk configuration.
    """

    parallel: bool  # Enable parallel computation with Dask
    chunk: ChunkParameterProtocol  # Chunk size configuration

    def to_dataset(self) -> xr.Dataset:
        """Convert all forcing fields to xarray Dataset.

        Returns:
            xr.Dataset: Dataset containing all forcing fields for the model.
        """
        ...


class FunctionalGroupParameterProtocol(Protocol):
    """Protocol for functional group parameter classes.

    Defines the interface for classes that manage collections of
    functional groups and their parameters.
    """

    functional_group: Iterable[FunctionalGroupUnitProtocol]  # Collection of functional groups

    def to_dataset(self, timestep: int) -> xr.Dataset:
        """Convert all functional groups to xarray Dataset.

        Args:
            timestep: Time step value for the conversion.

        Returns:
            xr.Dataset: Dataset containing all functional group parameters.
        """
        ...


# Phase 3 Protocols - Level 3 (depend on Phase 1 & 2 protocols)

class ConfigurationProtocol(Protocol):
    """Protocol for configuration classes.

    Defines the interface for classes that manage complete model configurations,
    including forcing data, functional groups, and kernel parameters.
    """

    forcing: ForcingParameterProtocol  # Forcing parameters
    functional_group: FunctionalGroupParameterProtocol  # Functional group parameters
    kernel: KernelParameterProtocol  # Kernel parameters

    @property
    def state(self) -> xr.Dataset:
        """Get the complete model state as xarray Dataset.

        Returns:
            xr.Dataset: Complete model state containing all parameters and forcing data.
                       This is the SeapopymState used throughout the model execution.
        """
        ...

    @classmethod
    def parse(cls, configuration_file: str | Path | IO) -> ConfigurationProtocol:
        """Parse configuration file and create a configuration object.

        Args:
            configuration_file: Path to configuration file or file-like object.

        Returns:
            ConfigurationProtocol: Parsed configuration object implementing this protocol.
        """
        ...


# Phase 4 Protocols - Level 4 (Model layer protocols)

class ModelProtocol(Protocol):
    """Protocol for model classes.

    Defines the interface for classes that represent complete computational models,
    including state management, kernel execution, and context management.
    """

    state: Any  # SeapopymState (xr.Dataset)
    kernel: Any  # Kernel object

    @classmethod
    def from_configuration(cls, configuration: ConfigurationProtocol) -> ModelProtocol:
        """Create a model from a configuration.

        Args:
            configuration: Configuration object implementing ConfigurationProtocol.

        Returns:
            ModelProtocol: Model instance created from the configuration.
        """
        ...

    def run(self) -> None:
        """Run the model.

        Executes the model's computational pipeline, updating the internal state.
        """
        ...

    def __enter__(self) -> ModelProtocol:
        """Enter context manager.

        Returns:
            ModelProtocol: Self for context manager protocol.
        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> None:
        """Exit context manager and cleanup memory.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_value: Exception value if an exception occurred.
            traceback: Traceback object if an exception occurred.
        """
        ...


# Phase 6 Protocols - Template layer protocols

class TemplateProtocol(Protocol):
    """Protocol for template classes.

    Defines the interface for classes that generate xarray templates
    for use with map_blocks operations in distributed computing.
    """

    def generate(self, state: Any) -> Any:
        """Generate an empty xr.DataArray or xr.Dataset template.

        Args:
            state: SeapopymState (xr.Dataset) containing model state and coordinates.

        Returns:
            Template structure as xr.DataArray or xr.Dataset for map_blocks operations.
        """
        ...
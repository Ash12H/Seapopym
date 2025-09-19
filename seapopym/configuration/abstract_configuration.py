"""Implementation of the base class for all configuration classes."""

from __future__ import annotations

import abc
from typing import IO, TYPE_CHECKING, Any

from attrs import define, field

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import xarray as xr

    from seapopym.standard.types import SeapopymState


@define
class AbstractForcingUnit:
    """Abstract class for a single forcing unit."""

    forcing: Any = field(metadata={"description": "The forcing field."})


class AbstractForcingParameter(abc.ABC):
    """
    Abstract class for forcing parameters.

    This class is intended to define and manage forcing parameters for a model.
    It can include attributes representing specific forcing units, which should
    be instances of `AbstractForcingUnit`.

    Example:
    -------
        A concrete implementation of this class might define attributes such as:

        ```python
        @attrs.define
        class MyForcingParameter(AbstractForcingParameter):
            temperature: MyForcingUnit
            oxygen: MyForcingUnit
        ```

    Subclasses must implement the `__attrs_post_init__` method to ensure
    consistency of units, timestep, and resolution.

    """

    parallel: bool = field(
        metadata={"description": "Enable parallel computation with Dask. Requires active Dask client."},
    )
    chunk: AbstractChunkParameter = field(
        metadata={"description": "The chunk size of the different dimensions for parallel computation."},
    )

    @abc.abstractmethod
    def to_dataset(self) -> xr.Dataset:
        """
        Return all the forcing fields as a xarray.Dataset.

        Can be implemented by child classes to add specific fields.
        """


# ParameterUnit class removed - replaced by direct use of pint.Quantity
# This eliminates the anti-pattern of inheriting from float and provides better
# scientific computation with proper unit handling via pint-xarray


# Former abstract classes removed - they were empty and only used for inheritance
# MigratoryTypeParameter and FunctionalTypeParameter now inherit directly from object


@define
class AbstractFunctionalGroupUnit(abc.ABC):
    """Abstract class for a single functional group unit."""

    name: str = field(metadata={"description": "The name of the functional group."})
    migratory_type: Any = field(
        metadata={"description": "The vertical migratory behavior of the functional group."}
    )
    functional_type: Any = field(
        metadata={
            "description": "Parameters related to the relationship between the functional group and its environment."
        }
    )

    @abc.abstractmethod
    def to_dataset(self: AbstractFunctionalGroupUnit) -> xr.Dataset:
        """Return the parameters of the functional group as a Dataset. This is used to create the SeapoPymState."""


@define
class AbstractFunctionalGroupParameter(abc.ABC):
    """Abstract class for functional group parameters."""

    functional_group: Iterable[AbstractFunctionalGroupUnit] = field(
        metadata={"description": "The functional groups of the model."}
    )

    @abc.abstractmethod
    def to_dataset(self: AbstractFunctionalGroupParameter) -> xr.Dataset:
        """Return all the functional groups as a xarray.Dataset."""


# AbstractClientParameter removed - it was never used anywhere in the codebase


@define
class AbstractChunkParameter(abc.ABC):
    """
    Abstract class for the chunk. Each attribute is a dimension of the state that can be splited into chunks to speed up
    the computation.
    """

    @abc.abstractmethod
    def as_dict(self) -> dict:
        """Format to a dictionary as expected by xarray."""


@define
class AbstractKernelParameter:
    """
    Abstract class for kernel parameters which are used to modify behaviour of kernel functions. These meta-parameters
    are integrated in the model state and used in kernel functions.
    """


@define
class AbstractConfiguration(abc.ABC):
    """Abstract class for all configurations."""

    forcing: AbstractForcingParameter = field(metadata={"description": "The forcing parameters for the configuration."})
    functional_group: AbstractFunctionalGroupParameter = field(
        metadata={"description": "The functional group parameters for the configuration."}
    )
    kernel: AbstractKernelParameter = field(metadata={"description": "The kernel parameters for the configuration."})

    @property
    @abc.abstractmethod
    def state(self: AbstractConfiguration) -> SeapopymState:
        """The xarray.Dataset that stores all the model parameters and forcing."""

    @classmethod
    @abc.abstractmethod
    def parse(cls, configuration_file: str | Path | IO) -> AbstractConfiguration:
        """Parse the configuration file and create an AbstractConfiguration object."""

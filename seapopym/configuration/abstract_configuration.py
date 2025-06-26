"""Implementation of the base class for all configuration classes."""

from __future__ import annotations

import abc
from typing import IO, TYPE_CHECKING, Any, Self

import pint
from attrs import define, field

if TYPE_CHECKING:
    from collections.abc import Iterable
    from numbers import Number
    from pathlib import Path

    import xarray as xr
    from dask.distributed import Client
    from pint import Unit

    from seapopym.standard.types import SeapopymState


@define
class AbstractForcingUnit(abc.ABC):
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

    timestep: Any = field(
        metadata={"description": "Simulations timestep."},
    )

    @abc.abstractmethod
    def to_dataset(self) -> xr.Dataset:
        """
        Return all the forcing fields as a xarray.Dataset.

        Can be implemented by child classes to add specific fields.
        """


@define
class ParameterUnit(float):
    """Abstract class describing a parameter. Derived from float. Defines a unit to avoid ambiguity."""

    value: float = field(metadata={"description": "The value of the functional type unit."})
    unit: str | Unit = field(metadata={"description": "The unit of the functional type unit."}, default="dimensionless")

    def __new__(cls, value: Number, unit: str | Unit = "dimensionless") -> Self:
        """Create a new instance of the ParameterUnit class derived from float."""
        instance = super().__new__(cls, value)
        instance.value = value
        instance.unit = unit
        return instance

    @property
    def quantity(self: ParameterUnit) -> pint.Quantity:
        """Return the value of the functional type unit as a pint.Quantity."""
        return self * pint.Unit(self.unit)

    def convert(self: ParameterUnit, unit: str | Unit) -> ParameterUnit:
        """Convert the value of the functional type unit to a different unit."""
        try:
            unit = pint.Unit(unit)
            quantity = self.quantity.to(unit)
        except pint.errors.UndefinedUnitError as e:
            msg = f"Unit {unit} is not defined in Pint."
            raise ValueError(msg) from e
        except pint.errors.DimensionalityError as e:
            msg = f"Cannot convert {self.unit} to {unit}."
            raise ValueError(msg) from e
        return ParameterUnit(quantity.magnitude, unit=unit)


@define
class AbstractMigratoryTypeParameter(abc.ABC):
    """Abstract class that describes the vertical migratory behavior of a functional group."""


@define
class AbstractFunctionalTypeParameter(abc.ABC):
    """Abstract class that describes the relationship between the functional group and its environment."""


@define
class AbstractFunctionalGroupUnit(abc.ABC):
    """Abstract class for a single functional group unit."""

    name: str = field(metadata={"description": "The name of the functional group."})
    migratory_type: AbstractMigratoryTypeParameter = field(
        metadata={"description": "The vertical migratory behavior of the functional group."}
    )
    functional_type: AbstractFunctionalTypeParameter = field(
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


@define
class AbstractClientParameter(abc.ABC):
    """Abstract class for a client."""

    client: Client | None = field(metadata={"description": "The Dask client."})


@define
class AbstractChunkParameter(abc.ABC):
    """
    Abstract class for the chunk. Each attribute is a dimension of the state that can be splited into chunks to speed up
    the computation.
    """


@define
class AbstractEnvironmentParameter(abc.ABC):
    """Abstract class for environment parameters."""

    chunk: AbstractChunkParameter = field(metadata={"description": "The chunk sizes."})


@define
class AbstractKernelParameter(abc.ABC):
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
    environment: AbstractEnvironmentParameter = field(
        metadata={"description": "The environment parameters for the configuration."}
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

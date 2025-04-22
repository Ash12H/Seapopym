"""This module define the data classes used to store the parameters of a functional group in the no transport model."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pint
from attrs import Attribute, field, frozen, validators

from seapopym.configuration.abstract_configuration import (
    AbstractFunctionalGroupParameter,
    AbstractFunctionalGroupUnit,
    AbstractFunctionalTypeParameter,
    AbstractMigratoryTypeParameter,
    FunctionalTypeUnit,
)
from seapopym.exception.parameter_exception import TimestepInDaysError
from seapopym.logging.custom_logger import logger

if TYPE_CHECKING:
    from collections.abc import Iterable


@frozen(kw_only=True)
class MigratoryTypeParameter(AbstractMigratoryTypeParameter):
    """This data class is used to store the parameters liked to the migratory behavior of a single functional group."""

    day_layer: int = field(
        converter=int,
        metadata={"description": "Layer position during day."},
    )
    night_layer: int = field(
        converter=int,
        metadata={"description": "Layer position during night."},
    )


def verify_unit(instance, attribute, value: FunctionalTypeUnit, parameter_name: str, unit: str) -> None:
    """
    This function is used to check if the unit of a parameter is correct.
    It raises a DimensionalityError if the unit is not correct.
    """
    try:
        value.quantity.to(unit)
    except pint.DimensionalityError as e:
        message = (
            f"Parameter {parameter_name} : {value} is not in the right unit. It should be convertible to {unit}. "
            f"Error: {e}"
        )
        raise pint.DimensionalityError(message) from e


@frozen(kw_only=True)
class FunctionalTypeParameter(AbstractFunctionalTypeParameter):
    """
    This data class is used to store the parameters linked to the relation between temperature and functional
    group.
    """

    lambda_0: FunctionalTypeUnit = field(
        converter=partial(FunctionalTypeUnit, unit="1/second"),
        validator=[
            partial(verify_unit, unit="1/second", parameter_name="lambda_0"),
            validators.ge(0),
        ],
        metadata={"description": "Value of lambda when temperature is 0°C."},
    )
    gamma_lambda: FunctionalTypeUnit = field(
        converter=partial(FunctionalTypeUnit, unit="1/degree_Celsius"),
        validator=[
            partial(verify_unit, unit="1/degree_Celsius", parameter_name="gamma_lambda"),
            validators.gt(0),
        ],
        metadata={"description": "Rate of the inverse of the mortality."},
    )
    tr_0: FunctionalTypeUnit = field(
        converter=partial(FunctionalTypeUnit, unit="second"),
        validator=[
            partial(verify_unit, unit="second", parameter_name="tr_0"),
            validators.ge(0),
        ],
        metadata={"description": "Maximum value of the recruitment age (i.e. when temperature is 0°C)."},
    )
    gamma_tr: FunctionalTypeUnit = field(
        converter=partial(FunctionalTypeUnit, unit="1/degree_Celsius"),
        validator=[
            partial(verify_unit, unit="1/degree_Celsius", parameter_name="gamma_tr"),
            validators.lt(0),
        ],
        metadata={"description": "Sensibility of recruitment age to temperature."},
    )
    # TODO(Jules): Automatically compute from tr_0 if None
    cohorts_timesteps: list[int] = field(
        metadata={"description": "The number of timesteps in the cohort. Useful for cohorts aggregation."},
    )

    @cohorts_timesteps.validator
    def _cohorts_timesteps_equal_tr_max(
        self: FunctionalTypeParameter, attribute: Attribute, value: Iterable[int]
    ) -> None:
        if not np.all(np.asarray(value) % 1 == 0):
            raise TimestepInDaysError(value)
        if np.sum(value) != np.ceil(self.tr_0):
            message = (
                f"Parameter {attribute.name} : {value} does not sum (= {np.sum(value)}) to the maximum recruitment "
                f"time {np.ceil(self.tr_0)} (ceiled value)."
            )
            raise ValueError(message)

    def __attrs_post_init__(self: FunctionalTypeParameter) -> None:
        """Ensure that the last cohort contains a single timestep."""
        if self.cohorts_timesteps[-1] != 1:
            previous = np.copy(self.cohorts_timesteps)
            new = np.copy(previous)
            new[-1] = new[-1] - 1
            new = np.concatenate([new, [1]])
            object.__setattr__(self, "cohorts_timesteps", new)
            msg = (
                "The last cohort timesteps must be equal to 1. It has been modified to follow the standard behavior."
                f"\nPrevious :{previous}\nNew : {new}"
            )
            logger.warning(msg)


@frozen(kw_only=True)
class FunctionalGroupUnit(AbstractFunctionalGroupUnit):
    """Represent a functional group."""

    name: str = field(metadata={"description": "The name of the functional group."})

    energy_transfert: float = field(
        validator=[validators.ge(0), validators.le(1)],
        converter=float,
        metadata={"description": "Energy transfert coefficient between primary production and functional group."},
    )

    functional_type: FunctionalTypeParameter = field(
        validator=validators.instance_of(FunctionalTypeParameter),
        metadata={"description": "Parameters linked to the relation between temperature and the functional group."},
    )

    migratory_type: MigratoryTypeParameter = field(
        validator=validators.instance_of(MigratoryTypeParameter),
        metadata={"description": "Parameters linked to the migratory behavior of the functional group."},
    )


@frozen(kw_only=True)
class FunctionalGroupParameter(AbstractFunctionalGroupParameter):
    """This data class is used to store the parameters of all functional groups."""

    functional_group: list[FunctionalGroupUnit] = field(metadata={"description": "List of all functional groups."})

    @functional_group.validator
    def are_all_instance_of_functional_group_unit(
        self: FunctionalGroupParameter, attribute: str, value: list[FunctionalGroupUnit]
    ) -> None:
        """This method is used to check the consistency of the functional groups."""
        if not all(isinstance(fgroup, FunctionalGroupUnit) for fgroup in value):
            msg = "All the functional groups must be instance of FunctionalGroupUnit."
            raise TypeError(msg)

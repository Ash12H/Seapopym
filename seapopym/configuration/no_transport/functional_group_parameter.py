"""This module define the data classes used to store the parameters of a functional group in the no transport model."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import cached_property, partial
from typing import TYPE_CHECKING

import numpy as np
import pint
import xarray as xr
from attrs import asdict, field, frozen, validators

from seapopym.configuration.abstract_configuration import (
    AbstractFunctionalGroupParameter,
    AbstractFunctionalGroupUnit,
    AbstractFunctionalTypeParameter,
    AbstractMigratoryTypeParameter,
    ParameterUnit,
)
from seapopym.standard.attributs import functional_group_desc
from seapopym.standard.coordinates import new_cohort
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels

if TYPE_CHECKING:
    from numbers import Number

logger = logging.getLogger(__name__)


@frozen(kw_only=True)
class MigratoryTypeParameter(AbstractMigratoryTypeParameter):
    """This data class is used to store the parameters liked to the migratory behavior of a single functional group."""

    day_layer: int = field(
        alias=ConfigurationLabels.day_layer,
        converter=int,
        metadata={"description": "Layer position during day."},
    )
    night_layer: int = field(
        alias=ConfigurationLabels.night_layer,
        converter=int,
        metadata={"description": "Layer position during night."},
    )


def verify_init(value: Number, unit: str | pint.Unit, parameter_name: str) -> ParameterUnit:
    """
    This function is used to check if the value of a parameter is correct.
    It raises a ValueError if the value is not correct.
    """
    if isinstance(value, ParameterUnit):
        try:
            new_value = value.quantity.to(unit).magnitude
        except pint.DimensionalityError as e:
            message = (
                f"Parameter {parameter_name} : {value} is not in the right unit. It should be convertible to {unit}. "
                f"Error: {e}"
            )
            raise ValueError(message) from e
        return ParameterUnit(new_value, unit=unit)
    return ParameterUnit(value, unit=unit)


@frozen(kw_only=True)
class FunctionalTypeParameter(AbstractFunctionalTypeParameter):
    """
    This data class is used to store the parameters linked to the relation between temperature and functional
    group.
    """

    lambda_temperature_0: ParameterUnit = field(
        alias=ConfigurationLabels.lambda_temperature_0,
        converter=partial(verify_init, unit="1/day", parameter_name=ConfigurationLabels.lambda_temperature_0),
        validator=validators.ge(0),
        metadata={"description": "Value of lambda when temperature is 0°C."},
    )
    gamma_lambda_temperature: ParameterUnit = field(
        alias=ConfigurationLabels.gamma_lambda_temperature,
        converter=partial(verify_init, unit="1/degC", parameter_name=ConfigurationLabels.gamma_lambda_temperature),
        validator=validators.gt(0),
        metadata={"description": "Rate of the inverse of the mortality."},
    )
    tr_0: ParameterUnit = field(
        alias=ConfigurationLabels.tr_0,
        converter=partial(verify_init, unit="day", parameter_name=ConfigurationLabels.tr_0),
        validator=validators.ge(0),
        metadata={"description": "Maximum value of the recruitment age (i.e. when temperature is 0°C)."},
    )
    gamma_tr: ParameterUnit = field(
        alias=ConfigurationLabels.gamma_tr,
        converter=partial(verify_init, unit="1/degC", parameter_name=ConfigurationLabels.gamma_tr),
        validator=validators.lt(0),
        metadata={"description": "Sensibility of recruitment age to temperature."},
    )


@frozen(kw_only=True)
class FunctionalGroupUnit(AbstractFunctionalGroupUnit):
    """Represent a functional group."""

    name: str = field(
        metadata={"description": "The name of the functional group."},
        alias=ConfigurationLabels.fgroup_name,
    )

    energy_transfert: ParameterUnit = field(
        alias=ConfigurationLabels.energy_transfert,
        converter=partial(verify_init, unit="dimensionless", parameter_name=ConfigurationLabels.energy_transfert),
        validator=[validators.ge(0), validators.le(1)],
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

    cohort_timestep: list[int] | None = field(
        default=None,
        metadata={
            "description": (
                "The number of timesteps in the cohort. Useful for cohorts aggregation. Last timestep must be 1."
            )
        },
    )

    @cohort_timestep.validator
    def check_cohort_timestep(self: FunctionalGroupUnit, attribute: str, value: list[int]) -> None:
        """Check that the last cohort is equal to 1."""
        if not isinstance(value, Iterable | None):
            msg = f"The {attribute.name} must be a list of integers or None."
            raise TypeError(msg)
        if value is not None and value[-1] != 1:
            msg = "The last cohort timestep must be equal to 1."
            raise ValueError(msg)

    def update_cohort_timestep(self: FunctionalGroupUnit, timestep: int) -> np.ndarray[int]:
        """
        This method updates the cohorts timesteps. The last cohort is always one timestep long and has an age equal
        to tr_0, which represents the maximum age of the pre-production in the coldest water conditions.
        """

        def initialize_cohort_timestep() -> np.ndarray[int]:
            max_age = self.functional_type.tr_0
            result = [1] * int(max_age // timestep)
            if max_age % timestep != 0:
                result.append(1)
            return np.asarray(result, dtype=int)

        def check_validity() -> np.ndarray[int]:
            cumsum_timesteps = np.cumsum(self.cohort_timestep) * timestep
            valid_mask = cumsum_timesteps < self.functional_type.tr_0
            if not valid_mask.any():
                msg = "No valid timesteps found. Check the input data."
                raise ValueError(msg)
            last_valid_timestep = cumsum_timesteps[valid_mask][-1]
            remaining_timesteps = (self.functional_type.tr_0 - last_valid_timestep) // timestep
            residual = (self.functional_type.tr_0 - last_valid_timestep) % timestep

            if residual == 0:
                to_concat = [remaining_timesteps - 1, 1] if remaining_timesteps > 1 else [1]
            elif remaining_timesteps == 0:
                to_concat = [1]
            else:
                to_concat = [remaining_timesteps, 1]

            valid_cohort_timestep = np.concatenate((np.asarray(self.cohort_timestep)[valid_mask], to_concat), dtype=int)

            if not np.array_equal(valid_cohort_timestep, self.cohort_timestep):
                message = (
                    f"The cohorts timesteps {self.cohort_timestep} are not valid. According to the values provided, "
                    f"the cohort_timestep is set to {valid_cohort_timestep}."
                )
                logger.warning(message)
            return valid_cohort_timestep

        if self.cohort_timestep is None:
            return initialize_cohort_timestep()
        return check_validity()

    def age_to_dataset(self: FunctionalGroupUnit, timestep: int) -> xr.Dataset:
        """
        Computes the mean, minimum, and maximum age of the cohorts at each timestep. The last cohort is always
        one timestep long and has an age equal to tr_0, which represents the maximum age of the pre-production
        in the coldest water conditions.
        """
        timesteps_number = self.update_cohort_timestep(timestep)

        cohort_index = new_cohort(np.arange(0, len(timesteps_number), 1, dtype=int))
        max_timestep = np.cumsum(timesteps_number)
        min_timestep = max_timestep - (np.asarray(timesteps_number) - 1)
        mean_timestep = (max_timestep + min_timestep) / 2

        data_vars = {
            ConfigurationLabels.timesteps_number: (
                (CoordinatesLabels.functional_group, CoordinatesLabels.cohort),
                np.asarray([timesteps_number]) * timestep,
                {
                    "description": (
                        "The number of timesteps represented in the cohort. If there is no aggregation, all values are "
                        "equal to 1."
                    ),
                    "units": "day",
                },
            ),
            ConfigurationLabels.min_timestep: (
                (CoordinatesLabels.functional_group, CoordinatesLabels.cohort),
                np.asarray([min_timestep]) * timestep,
                {"description": "The minimum timestep index.", "units": "day"},
            ),
            ConfigurationLabels.max_timestep: (
                (CoordinatesLabels.functional_group, CoordinatesLabels.cohort),
                np.asarray([max_timestep]) * timestep,
                {"description": "The maximum timestep index.", "units": "day"},
            ),
            ConfigurationLabels.mean_timestep: (
                (CoordinatesLabels.functional_group, CoordinatesLabels.cohort),
                np.asarray([mean_timestep]) * timestep,
                {"description": "The mean timestep index.", "units": "day"},
            ),
        }

        return xr.Dataset(
            coords={CoordinatesLabels.cohort: cohort_index},
            data_vars=data_vars,
        )

    @cached_property
    def parameter_to_dataset(self) -> xr.Dataset:
        """Return the parameters of the functional group as a Dataset. This is used to create the SeapoPymState."""
        parameters = {
            ConfigurationLabels.fgroup_name: self.name,
            ConfigurationLabels.energy_transfert: self.energy_transfert,
            **asdict(self.functional_type, recurse=False),
            **asdict(self.migratory_type, recurse=False),
        }
        return xr.Dataset(
            {k: v.value * pint.Unit(v.unit) if isinstance(v, ParameterUnit) else v for k, v in parameters.items()}
        )

    def to_dataset(self: FunctionalGroupUnit, timestep: int) -> xr.Dataset:
        """
        This method is used to create the dataset of the functional group. It contains the parameters of the
        functional group and the age of the cohorts.
        """
        return xr.merge([self.parameter_to_dataset, self.age_to_dataset(timestep)])


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

    def to_dataset(self: FunctionalGroupParameter, timestep: int) -> xr.Dataset:
        all_dataset = [fgroup.to_dataset(timestep) for fgroup in self.functional_group]

        coordinates = xr.DataArray(
            data=range(len(all_dataset)),
            dims=[CoordinatesLabels.functional_group],
            attrs=functional_group_desc(range(len(all_dataset)), [fgroup.name for fgroup in self.functional_group]),
        )

        return xr.concat(all_dataset, dim=coordinates)

"""Define the ForcingUnit data class used to store access paths to a forcing field."""

from __future__ import annotations

import logging
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, ParamSpecArgs, ParamSpecKwargs

import cf_xarray  # noqa: F401
import fsspec
import pandas as pd
import pint
import xarray as xr
from attrs import asdict, converters, field, frozen, validators
from pandas.tseries.frequencies import to_offset

from seapopym.configuration.abstract_configuration import AbstractForcingParameter, AbstractForcingUnit
from seapopym.standard.labels import ConfigurationLabels, ForcingLabels
from seapopym.standard.units import StandardUnitsLabels

if TYPE_CHECKING:
    from pint import Unit

logger = logging.getLogger(__name__)

DECIMALS = 5  # ie. 1e-5 degrees which is equivalent to ~1m at the equator


def path_validation(path: str | Path) -> str | Path:
    """Check if the path exists."""
    with fsspec.open(str(path)) as file:
        if "file" not in file.fs.protocol:
            message = f"Remote file : {file.fs.protocol}"
            logger.info(message)
            return str(path)
        if "file" in file.fs.protocol and Path(path).exists():
            message = f"Local file : ({file.fs.protocol})"
            logger.debug(message)
            return Path(path)
    msg = f"Cannot reach '{path}'."
    raise FileNotFoundError(msg)


@frozen(kw_only=True)
class ForcingUnit(AbstractForcingUnit):
    """
    This data class is used to store a forcing field.

    Parameters
    ----------
    forcing: xr.DataArray
        Forcing field.


    Notes
    -----
    - This class is used to store a forcing field.
    - Be sure to follow the CF conventions for the forcing file. To do so you can use the `cf_xarray` package.

    """

    forcing: xr.DataArray = field(
        converter=xr.DataArray,
        metadata={"description": "Forcing field."},
    )

    @classmethod
    def from_dataset(
        cls: ForcingUnit,
        forcing: xr.Dataset,
        name: str,
    ) -> ForcingUnit:
        """Create a ForcingUnit from a path and a name."""
        if name not in forcing:
            message = f"DataArray {name} is not in the Dataset.\nAccepted values are : {', '.join(list(forcing))}"
            raise ValueError(message)
        return cls(forcing=forcing[name])

    @classmethod
    def from_path(
        cls: ForcingUnit,
        forcing: Path | str,
        name: str,
        engine: Literal["zarr", "netcdf"] = "zarr",
        *args: ParamSpecArgs,
        **kwargs: ParamSpecKwargs,
    ) -> ForcingUnit:
        """Create a ForcingUnit from a path and a name."""
        path_validation(forcing)
        data = xr.open_dataset(forcing, *args, engine=engine, **kwargs)
        return cls.from_dataset(data, name)

    def convert(self: ForcingUnit, units: str | Unit) -> ForcingUnit:
        """
        Create a new ForcingUnit with the same forcing field but with a different unit.

        Parameters.
        ----------
        units: str | Unit
            The unit to convert the forcing field to. If a string is provided, it will be converted to a Pint Unit.
            If a Pint Unit is provided, it will be used as is.
        """
        try:
            if isinstance(units, str):
                units = pint.Unit(units)
        except pint.errors.UndefinedUnitError as e:
            msg = f"Unit {units} is not defined in Pint."
            raise ValueError(msg) from e

        if self.forcing.pint.units is None:
            try:
                forcing = self.forcing.pint.quantify()
            except pint.errors.DimensionalityError as e:
                message = f"Cannot quantify {self.forcing.name} because it has no units."
                raise ValueError(message) from e

        if forcing.pint.units != units:
            message = f"{forcing.name} unit is {forcing.pint.units}, it will be converted to {units}."
            logger.warning(message)
        try:
            forcing = forcing.pint.to(units)
        except Exception as e:
            message = f"Failed to convert forcing to {units}. forcing is in {forcing.pint.units}."
            logger.exception(message)
            raise type(e)(message) from e

        return type(self)(forcing=forcing.pint.dequantify())


def verify_init(value: ForcingUnit, unit: str | Unit, parameter_name: str) -> ForcingUnit:
    """
    This function is used to check if the unit of a parameter is correct.
    It raises a DimensionalityError if the unit is not correct.
    """
    value.convert(unit)
    if isinstance(value, ForcingUnit):
        return value.convert(unit)
    return ForcingUnit(value)


@frozen(kw_only=True)
class ForcingParameter(AbstractForcingParameter):
    """
    This data class is used to store access paths to forcing fields. You can inherit it to add further forcings, but in
    this case you'll need to add new behaviors to the functions and classes that follow.
    """

    temperature: ForcingUnit = field(
        alias=ForcingLabels.temperature,
        converter=partial(
            verify_init, unit=StandardUnitsLabels.temperature.units, parameter_name=ForcingLabels.temperature
        ),
        validator=validators.instance_of(ForcingUnit),
        metadata={"description": "Path to the temperature field."},
    )
    primary_production: ForcingUnit = field(
        alias=ForcingLabels.primary_production,
        converter=partial(
            verify_init, unit=StandardUnitsLabels.production.units, parameter_name=ForcingLabels.primary_production
        ),
        validator=validators.instance_of(ForcingUnit),
        metadata={"description": "Path to the primary production field."},
    )

    initial_condition_production: ForcingUnit | None = field(
        alias=ConfigurationLabels.initial_condition_production,
        default=None,
        converter=converters.optional(
            partial(
                verify_init,
                unit=StandardUnitsLabels.production.units,
                parameter_name=ConfigurationLabels.initial_condition_production,
            )
        ),
        validator=validators.optional(validators.instance_of(ForcingUnit)),
        metadata={"description": "Path to the initial condition production field.", "dims": "Fgroup, <Y, X,> Cohort"},
    )

    initial_condition_biomass: ForcingUnit | None = field(
        alias=ConfigurationLabels.initial_condition_biomass,
        default=None,
        converter=converters.optional(
            partial(
                verify_init,
                unit=StandardUnitsLabels.biomass.units,
                parameter_name=ConfigurationLabels.initial_condition_biomass,
            )
        ),
        validator=validators.optional(validators.instance_of(ForcingUnit)),
        metadata={"description": "Path to the initial condition biomass field.", "dims": "Fgroup, <Y, X>"},
    )

    def __attrs_post_init__(self: ForcingParameter) -> None:
        """Post initialization to ensure all forcing fields are valid."""
        # 1. Check timestep consistency
        timestep = self.to_dataset().cf.indexes["T"].to_series().diff().dt.days.dropna().unique()
        if len(timestep) != 1:
            msg = (
                f"Expected a single unique timestep in the dataset, found {len(timestep)} unique values: {timestep}.\n"
                "Ensure that all forcing fields have the same time resolution."
            )
            raise ValueError(msg)

        # 2. Check nans consistency
        for name, forcing in self.all_forcings.items():
            if "T" in forcing.forcing.cf:
                valid_counts: xr.DataArray = forcing.forcing.notnull().cf.sum(dim="T")
                total_timesteps = forcing.forcing.cf.sizes["T"]
                inconsistent_cells = (valid_counts > 0) & (valid_counts < total_timesteps)
                if inconsistent_cells.any():
                    message = (
                        f"Warning: {name} has cells with inconsistent NaN patterns across time. These cells have valid "
                        "values for some timesteps but NaN for others. This may cause issues with global mask "
                        "generation."
                    )
                    logger.warning(message)

    @property
    def all_forcings(self: ForcingParameter) -> dict[str, ForcingUnit]:
        """Return all the not null ForcingUnit as a dictionary."""
        return asdict(self, recurse=False, filter=lambda _, value: isinstance(value, ForcingUnit))

    def to_dataset(self) -> xr.Dataset:
        """An xarray.Dataset containing all the forcing fields used to construct the SeapoPymState."""
        return xr.Dataset({k: v.forcing for k, v in self.all_forcings.items() if v.forcing is not None})

"""Define the ForcingUnit data class used to store access paths to a forcing field."""

from __future__ import annotations

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
from seapopym.logging.custom_logger import logger
from seapopym.standard.labels import ConfigurationLabels, ForcingLabels
from seapopym.standard.units import StandardUnitsLabels

if TYPE_CHECKING:
    from pint import Unit

DECIMALS = 5  # ie. 1e-5 degrees which is equivalent to ~1m at the equator


def path_validation(path: str | Path) -> str | Path:
    """Check if the path exists."""
    with fsspec.open(str(path)) as file:
        if "file" not in file.fs.protocol:
            logger.info(f"Remote file : {file.fs.protocol}")
            return str(path)
        if "file" in file.fs.protocol and Path(path).exists():
            # logger.debug(f"Local file : ({file.fs.protocol})")
            return Path(path)
    msg = f"Cannot reach '{path}'."
    raise FileNotFoundError(msg)


@frozen(kw_only=True)
class ForcingUnit(AbstractForcingUnit):
    """
    This data class is used to store a forcing field and its resolution and timestep.

    Parameters
    ----------
    forcing: xr.DataArray
        Forcing field.
    resolution: tuple[float, float] | None
        Space resolution of the field as (lat, lon).
    timestep: int | None
        Timestep of the field in day(s).


    Notes
    -----
    - This class is used to store a forcing field. It also stores the resolution and timestep of the field. If not
    provided, the resolution and timestep are automatically computed from the forcing file. However, they can be set
    manually.
    - Be sure to follow the CF conventions for the forcing file. To do so you can use the `cf_xarray` package.

    """

    forcing: xr.DataArray = field(
        converter=xr.DataArray,
        metadata={"description": "Forcing field."},
    )

    # NOTE(Jules):  For resolution and timestep, `default=None` because these attributes are automatically computed from
    #               the forcing file. However, they can be set manually.

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
            logger.warning(f"{forcing.name} unit is {forcing.pint.units}, it will be converted to {units}.")
        try:
            forcing = forcing.pint.to(units)
        except Exception as e:
            message = f"Failed to convert forcing to {units}. forcing is in {forcing.pint.units}."
            logger.error(message)
            raise type(e)(message) from e

        return type(self)(forcing=forcing)


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
    global_mask: ForcingUnit | None = field(
        alias=ForcingLabels.global_mask,
        default=None,
        validator=validators.optional(validators.instance_of(ForcingUnit)),
        metadata={"description": "Path to the global_mask field."},
    )
    day_length: ForcingUnit | None = field(
        alias=ForcingLabels.day_length,
        default=None,
        converter=converters.optional(
            partial(verify_init, unit=StandardUnitsLabels.time.units, parameter_name=ForcingLabels.day_length)
        ),
        validator=validators.optional(validators.instance_of(ForcingUnit)),
        metadata={"description": "Path to the day length field."},
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

    timestep: pd.offsets.BaseOffset = field(
        converter=to_offset,
        default=pd.offsets.Day(1),
        validator=validators.instance_of(pd.offsets.BaseOffset),
        metadata={"description": ("Simulation timesteps expressed as a pandas offset.")},
    )

    @property
    def all_forcings(self: ForcingParameter) -> dict[str, ForcingUnit]:
        """Return all the not null ForcingUnit as a dictionary."""
        return asdict(self, recurse=False, filter=lambda _, value: isinstance(value, ForcingUnit))

    @cached_property
    def to_dataset(self) -> xr.Dataset:
        """An xarray.Dataset containing all the forcing fields used to construct the SeapoPymState."""

        def resample_to_timestep(forcing: xr.DataArray) -> xr.DataArray:
            return (
                forcing.pint.dequantify()
                .cf.resample({"T": self.timestep})
                .mean()
                .cf.interpolate_na("T")
                .cf.dropna(dim="T", how="any")
            )

        forcing = {
            k: resample_to_timestep(v.forcing) if "T" in v.forcing.cf.indexes else v.forcing
            for k, v in self.all_forcings.items()
            if v.forcing is not None
        }
        return xr.Dataset(forcing)

    def timestep_in_day(self: ForcingParameter) -> int:
        """Return the timestep in days."""
        timestep = self.to_dataset.cf.indexes["T"].to_series().diff().dt.days.dropna().unique()
        if len(timestep) != 1:
            msg = (
                f"Cannot determine timestep in days. Found {timestep} instead. If you are using non daily data, please "
                "ensure you are using the right calendar. It is highly recommended to use the CF calendar using 360_day"
                " for monthly data."
            )
            raise ValueError(msg)
        return int(timestep[0])

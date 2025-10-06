"""Define the ForcingUnit data class used to store access paths to a forcing field."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, ParamSpecArgs, ParamSpecKwargs

import cf_xarray  # noqa: F401
import fsspec
import numpy as np
import pint
import xarray as xr
from attrs import asdict, converters, field, frozen, validators

from seapopym.configuration.validation import validate_coordinate_coherence, verify_forcing_init
from seapopym.standard.labels import ConfigurationLabels, ForcingLabels
from seapopym.standard.units import StandardUnitsLabels

if TYPE_CHECKING:
    from pint import Unit

logger = logging.getLogger(__name__)

DECIMALS = 5  # ie. 1e-5 degrees which is equivalent to ~1m at the equator


@frozen
class ChunkParameter:
    """The chunk size of the different dimensions."""

    functional_group: Literal["auto"] | int | None = field(
        default=1,
        validator=validators.optional(validators.instance_of((str, int))),
        metadata={"description": "The chunk size of the functional group dimension."},
    )
    Y: Literal["auto"] | int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of((str, int))),
        metadata={"description": "The chunk size of the Y (latitude) dimension."},
    )
    X: Literal["auto"] | int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of((str, int))),
        metadata={"description": "The chunk size of the X (longitude) dimension."},
    )
    T: Literal[-1] = field(
        init=False,
        default=-1,
        metadata={
            "description": (
                "The chunk size of the T (time) dimension. "
                "Present only to remind us that time is not divisible due to time dependencies."
            )
        },
    )

    def as_dict(self: ChunkParameter) -> dict:
        """Format to a dictionary as expected by xarray with standardized coordinates."""
        chunks = {}
        if self.functional_group is not None:
            chunks["functional_group"] = self.functional_group
        if self.Y is not None:
            chunks["Y"] = self.Y
        if self.X is not None:
            chunks["X"] = self.X
        chunks["T"] = self.T
        return chunks


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
class ForcingUnit:
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

    def __attrs_post_init__(self) -> None:
        """Apply coordinate standardization after initialization."""
        standardized_forcing = self._standardize_coordinates(self.forcing)
        # Use object.__setattr__ because @frozen prevents normal assignment
        object.__setattr__(self, "forcing", standardized_forcing)

    @classmethod
    def from_dataset(
        cls: ForcingUnit,
        forcing: xr.Dataset,
        name: str,
    ) -> ForcingUnit:
        """Create a ForcingUnit from a path and a name with standardized coordinate names."""
        if name not in forcing:
            message = f"DataArray {name} is not in the Dataset.\nAccepted values are : {', '.join(list(forcing))}"
            raise ValueError(message)

        data_array = forcing[name]
        # La standardisation se fait automatiquement dans __attrs_post_init__
        return cls(forcing=data_array)

    @staticmethod
    def _standardize_coordinates(data_array: xr.DataArray) -> xr.DataArray:
        """Rename coordinates to T/Y/X/Z using cf_xarray, keep attributes."""
        coord_mapping = {}

        try:
            if "T" in data_array.cf:
                original_time = data_array.cf["T"].name
                if original_time != "T":
                    coord_mapping[original_time] = "T"

            if "Y" in data_array.cf:
                original_lat = data_array.cf["Y"].name
                if original_lat != "Y":
                    coord_mapping[original_lat] = "Y"

            if "X" in data_array.cf:
                original_lon = data_array.cf["X"].name
                if original_lon != "X":
                    coord_mapping[original_lon] = "X"

            if "Z" in data_array.cf:
                original_z = data_array.cf["Z"].name
                if original_z != "Z":
                    coord_mapping[original_z] = "Z"

        except Exception as e:
            logger.warning(f"Could not standardize coordinates using cf_xarray: {e}. Keeping original names.")
            return data_array

        if coord_mapping:
            logger.info(f"Standardizing coordinates: {coord_mapping}")
            return data_array.rename(coord_mapping)

        return data_array

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


# verify_init function moved to seapopym.configuration.validation module
# Use verify_forcing_init instead


@frozen(kw_only=True)
class ForcingParameter:
    """
    This data class is used to store access paths to forcing fields. You can inherit it to add further forcings, but in
    this case you'll need to add new behaviors to the functions and classes that follow.
    """

    temperature: ForcingUnit = field(
        alias=ForcingLabels.temperature,
        converter=partial(
            verify_forcing_init, unit=StandardUnitsLabels.temperature.units, parameter_name=ForcingLabels.temperature
        ),
        validator=validators.instance_of(ForcingUnit),
        metadata={"description": "Path to the temperature field."},
    )
    primary_production: ForcingUnit = field(
        alias=ForcingLabels.primary_production,
        converter=partial(
            verify_forcing_init,
            unit=StandardUnitsLabels.production.units,
            parameter_name=ForcingLabels.primary_production,
        ),
        validator=validators.instance_of(ForcingUnit),
        metadata={"description": "Path to the primary production field."},
    )

    initial_condition_production: ForcingUnit | None = field(
        alias=ConfigurationLabels.initial_condition_production,
        default=None,
        converter=converters.optional(
            partial(
                verify_forcing_init,
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
                verify_forcing_init,
                unit=StandardUnitsLabels.biomass.units,
                parameter_name=ConfigurationLabels.initial_condition_biomass,
            )
        ),
        validator=validators.optional(validators.instance_of(ForcingUnit)),
        metadata={"description": "Path to the initial condition biomass field.", "dims": "Fgroup, <Y, X>"},
    )

    parallel: bool = field(
        default=False,
        validator=validators.instance_of(bool),
        metadata={"description": "Enable parallel computation with Dask. Requires active Dask client."},
    )

    chunk: ChunkParameter = field(
        factory=ChunkParameter,
        validator=validators.instance_of(ChunkParameter),
        metadata={"description": "The chunk size of the different dimensions for parallel computation."},
    )

    def __attrs_post_init__(self: ForcingParameter) -> None:
        """Post initialization with flexible coherence validation."""
        # 0. Check parallel computation setup
        if self.parallel:
            self._validate_dask_client()
        self._validate_forcing_consistency()

        # 1. Validation de cohérence flexible (remplace validation timestep basique)
        validate_coordinate_coherence(self.all_forcings)

        # 2. Check nans consistency
        self._validate_nan_consistency()

    def _validate_nan_consistency(self: ForcingParameter) -> None:
        """Validate NaN consistency across time for all forcings."""
        for name, forcing in self.all_forcings.items():
            if "T" in forcing.forcing.coords:
                valid_counts: xr.DataArray = forcing.forcing.notnull().sum(dim="T")
                total_timesteps = forcing.forcing.sizes["T"]
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

    def _validate_dask_client(self: ForcingParameter) -> None:
        """Ensure Dask client is available for parallel computation."""
        try:
            from dask.distributed import get_client

            get_client()
            logger.info("Dask client found, parallel computation enabled.")
        except (ImportError, ValueError) as e:
            msg = (
                "parallel=True requires an active Dask client. "
                "Start a client with: from dask.distributed import Client; client = Client()"
            )
            raise RuntimeError(msg) from e

    def _validate_forcing_consistency(self: ForcingParameter) -> None:
        """Validate consistency between parallel setting and forcing memory status."""
        dask_arrays = [hasattr(forcing.forcing.data, "chunks") for forcing in self.all_forcings.values()]

        if self.parallel and not all(dask_arrays):
            msg = (
                "parallel=True but forcings are loaded in memory (numpy arrays). "
                "For distributed computation, use chunked loading. Example:\n"
                "  forcing = xr.open_dataset('file.nc', chunks={'T': -1, 'Y': 180})\n"
                "Or scatter existing arrays:\n"
                "  from dask.distributed import get_client\n"
                "  client = get_client()\n"
                "  scattered_forcing = client.scatter(forcing, broadcast=True)"
            )
            raise ValueError(msg)

        if not self.parallel and any(dask_arrays):
            msg = (
                "parallel=False but forcings contain Dask arrays (lazy loading). "
                "For non-parallel computation, load forcings into memory:\n"
                "  forcing = forcing.load()  # or .compute()\n"
                "Or enable parallel computation:\n"
                "  parallel=True"
            )
            raise ValueError(msg)

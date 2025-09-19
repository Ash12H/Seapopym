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

from seapopym.configuration.abstract_configuration import (
    AbstractChunkParameter,
    AbstractForcingParameter,
    AbstractForcingUnit,
)
from seapopym.configuration.validation import verify_forcing_init
from seapopym.standard.labels import ConfigurationLabels, ForcingLabels
from seapopym.standard.units import StandardUnitsLabels

if TYPE_CHECKING:
    from pint import Unit

logger = logging.getLogger(__name__)

DECIMALS = 5  # ie. 1e-5 degrees which is equivalent to ~1m at the equator


@frozen
class ChunkParameter(AbstractChunkParameter):
    """The chunk size of the different dimensions."""

    functional_group: Literal["auto"] | int | None = field(
        default=1,
        validator=validators.optional(validators.instance_of((str, int))),
        metadata={"description": "The chunk size of the functional group dimension."},
    )
    latitude: Literal["auto"] | int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of((str, int))),
        metadata={"description": "The chunk size of the latitude dimension."},
    )
    longitude: Literal["auto"] | int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of((str, int))),
        metadata={"description": "The chunk size of the longitude dimension."},
    )
    time: Literal[-1] = field(
        init=False,
        default=-1,
        metadata={
            "description": (
                "The chunk size of the time dimension. "
                "Present only to remind us that time is not divisible due to time dependencies."
            )
        },
    )

    def as_dict(self: ChunkParameter) -> dict:
        """Format to a dictionary as expected by xarray."""
        chunks = {}
        if self.functional_group is not None:
            chunks["functional_group"] = self.functional_group
        if self.latitude is not None:
            chunks["latitude"] = self.latitude
        if self.longitude is not None:
            chunks["longitude"] = self.longitude
        chunks["time"] = self.time
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

    def __attrs_post_init__(self) -> None:
        """Apply coordinate standardization after initialization."""
        standardized_forcing = self._standardize_coordinates(self.forcing)
        # Use object.__setattr__ because @frozen prevents normal assignment
        object.__setattr__(self, 'forcing', standardized_forcing)

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
            # Détecter et mapper avec cf_xarray
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


class ForcingCoherenceValidator:
    """Validate coherence between forcing fields with standardized coordinates."""

    def __init__(self, forcings: dict[str, ForcingUnit]):
        self.forcings = forcings

    def validate_temporal_coherence(self) -> None:
        """Validate T coordinate coherence between ALL forcings that have T."""
        forcings_with_time = {
            name: forcing for name, forcing in self.forcings.items()
            if 'T' in forcing.forcing.coords
        }

        if len(forcings_with_time) < 2:
            return  # Pas assez de forçages temporels pour comparer

        self._validate_coordinate_coherence(forcings_with_time, 'T')

    def validate_spatial_coherence(self) -> None:
        """Validate X,Y coordinate coherence between ALL forcings that have spatial dims."""
        # Grouper par combinaisons de coordonnées spatiales
        spatial_groups = {}

        for name, forcing in self.forcings.items():
            spatial_dims = tuple(sorted([
                dim for dim in ['X', 'Y']
                if dim in forcing.forcing.coords
            ]))

            if spatial_dims:  # A au moins une coordonnée spatiale
                if spatial_dims not in spatial_groups:
                    spatial_groups[spatial_dims] = {}
                spatial_groups[spatial_dims][name] = forcing

        # Valider la cohérence dans chaque groupe
        for dims, group_forcings in spatial_groups.items():
            if len(group_forcings) > 1:
                self._validate_coordinate_coherence(group_forcings, dims)

    def validate_all_coherence(self) -> None:
        """Validate all possible coherence without assumptions about required fields."""
        # 1. Validation temporelle - tous les forçages avec T
        self.validate_temporal_coherence()

        # 2. Validation spatiale - tous les forçages avec X et/ou Y
        self.validate_spatial_coherence()

        # 3. Si présente, validation Z (optionnelle pour l'avenir)
        forcings_with_z = {
            name: forcing for name, forcing in self.forcings.items()
            if 'Z' in forcing.forcing.coords
        }
        if len(forcings_with_z) > 1:
            self._validate_coordinate_coherence(forcings_with_z, 'Z')

    def _validate_coordinate_coherence(self, forcings: dict[str, ForcingUnit], coords: str | tuple[str]) -> None:
        """Generic coordinate coherence validation."""
        if isinstance(coords, str):
            coords = (coords,)

        reference_name, reference_forcing = next(iter(forcings.items()))
        reference_coords = {coord: reference_forcing.forcing.coords[coord] for coord in coords}

        for name, forcing in forcings.items():
            if name == reference_name:
                continue

            forcing_coords = {coord: forcing.forcing.coords[coord] for coord in coords}

            if not self._are_coordinates_coherent(reference_coords, forcing_coords):
                coord_desc = '+'.join(coords)
                error_details = self._get_coherence_error_details(reference_coords, forcing_coords, coords)
                raise ValueError(
                    f"Coordinate incoherence ({coord_desc}) between '{reference_name}' and '{name}':\n{error_details}"
                )

    def _are_coordinates_coherent(self, coords1: dict[str, xr.DataArray], coords2: dict[str, xr.DataArray]) -> bool:
        """Check if two sets of coordinates are coherent."""
        for coord_name in coords1.keys():
            if coord_name not in coords2:
                return False

            coord1 = coords1[coord_name]
            coord2 = coords2[coord_name]

            # Vérifier que les tailles sont identiques
            if coord1.size != coord2.size:
                return False

            # Pour les coordonnées temporelles, vérifier les valeurs
            if coord_name == 'T':
                if not self._are_times_coherent(coord1, coord2):
                    return False

            # Pour les coordonnées spatiales, vérifier les valeurs avec tolérance
            elif coord_name in ['X', 'Y', 'Z']:
                if not self._are_spatial_coords_coherent(coord1, coord2):
                    return False

        return True

    def _are_times_coherent(self, time1: xr.DataArray, time2: xr.DataArray) -> bool:
        """Check if two time coordinates are coherent."""
        try:
            # Convertir en timestamps si nécessaire
            if hasattr(time1.values[0], 'timestamp'):
                times1 = [t.timestamp() for t in time1.values]
                times2 = [t.timestamp() for t in time2.values]
            else:
                times1 = time1.values
                times2 = time2.values

            return np.array_equal(times1, times2)
        except Exception:
            # Fallback: comparaison directe
            return np.array_equal(time1.values, time2.values)

    def _are_spatial_coords_coherent(self, coord1: xr.DataArray, coord2: xr.DataArray) -> bool:
        """Check if two spatial coordinates are coherent (with tolerance)."""
        try:
            return np.allclose(coord1.values, coord2.values, rtol=1e-5, atol=1e-5)
        except Exception:
            return np.array_equal(coord1.values, coord2.values)

    def _get_coherence_error_details(self, coords1: dict[str, xr.DataArray], coords2: dict[str, xr.DataArray], coord_names: tuple[str]) -> str:
        """Generate detailed error message for coordinate incoherence."""
        details = []

        for coord_name in coord_names:
            coord1 = coords1.get(coord_name)
            coord2 = coords2.get(coord_name)

            if coord1 is None or coord2 is None:
                details.append(f"  {coord_name}: Missing coordinate in one of the forcings")
                continue

            if coord1.size != coord2.size:
                details.append(f"  {coord_name}: Different sizes ({coord1.size} vs {coord2.size})")
                continue

            if coord_name == 'T':
                # Pour le temps, montrer la plage temporelle
                try:
                    time1_range = f"{coord1.values[0]} to {coord1.values[-1]}"
                    time2_range = f"{coord2.values[0]} to {coord2.values[-1]}"
                    details.append(f"  {coord_name}: Different time ranges\n    Reference: {time1_range}\n    Compared:  {time2_range}")
                except Exception:
                    details.append(f"  {coord_name}: Different time values")

            elif coord_name in ['X', 'Y', 'Z']:
                # Pour l'espace, montrer les bornes et la résolution
                try:
                    min1, max1 = float(coord1.min()), float(coord1.max())
                    min2, max2 = float(coord2.min()), float(coord2.max())
                    resolution1 = float(coord1.diff(coord1.dims[0]).mean()) if coord1.size > 1 else 0
                    resolution2 = float(coord2.diff(coord2.dims[0]).mean()) if coord2.size > 1 else 0

                    details.append(
                        f"  {coord_name}: Different spatial coordinates\n"
                        f"    Reference: {min1:.5f} to {max1:.5f} (res: {resolution1:.5f})\n"
                        f"    Compared:  {min2:.5f} to {max2:.5f} (res: {resolution2:.5f})"
                    )
                except Exception:
                    details.append(f"  {coord_name}: Different spatial values")

        return '\n'.join(details) if details else "  Unknown coordinate difference"


@frozen(kw_only=True)
class ForcingParameter(AbstractForcingParameter):
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
            verify_forcing_init, unit=StandardUnitsLabels.production.units, parameter_name=ForcingLabels.primary_production
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
        validator = ForcingCoherenceValidator(self.all_forcings)
        validator.validate_all_coherence()

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
                "  forcing = xr.open_dataset('file.nc', chunks={'time': -1, 'latitude': 180})\n"
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

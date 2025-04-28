"""Store all labels used in the No Transport model."""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import xarray as xr


class CoordinatesLabels(StrEnum):
    """A single place to store all labels as declared in coordinates module. It follow the cf_xarray convention."""

    functional_group = "functional_group"
    time = "T"
    Y = "Y"
    X = "X"
    Z = "Z"
    cohort = "cohort"

    @classmethod
    def ordered(cls: CoordinatesLabels) -> tuple[CoordinatesLabels]:
        """Return all labels in the order they should be used in a dataset. It follow the CF convention."""
        return (cls.functional_group, cls.time, cls.Y, cls.X, cls.Z, cls.cohort)

    @classmethod
    def order_data(cls: CoordinatesLabels, data: xr.Dataset | xr.DataArray) -> xr.Dataset:
        """Return the dataset with the coordinates ordered as in the CF convention."""
        return data.cf.transpose(*cls.ordered(), missing_dims="ignore")


class SeaLayers(Enum):
    """Enumerate the sea layers."""

    # NOTE(Jules): The following order of the layers declaration is important.
    ## Since python 3.4 this order is preserved.
    EPI = ("epipelagic", 1)
    UPMESO = ("upper-mesopelagic", 2)
    LOWMESO = ("lower-mesopelagic", 3)

    @property
    def standard_name(
        self: SeaLayers,
    ) -> Literal["epipelagic", "upper-mesopelagic", "lower-mesopelagic"]:
        """Return the standard_name of the sea layer."""
        return self.value[0]

    @property
    def depth(self: SeaLayers) -> Literal[1, 2, 3]:
        """Return the depth of the sea layer."""
        return self.value[1]


class ConfigurationLabels(StrEnum):
    """A single place to store all labels as declared in parameters module."""

    # Functional group
    fgroup_name = "name"
    energy_transfert = "energy_transfert"
    lambda_temperature_0 = "lambda_temperature_0"
    gamma_lambda_temperature = "gamma_lambda_temperature"
    lambda_acidity_0 = "lambda_acidity_0"
    gamma_lambda_acidity = "gamma_lambda_acidity"
    tr_0 = "tr_0"
    gamma_tr = "gamma_tr"
    day_layer = "day_layer"
    night_layer = "night_layer"
    # Cohorts
    cohort = "cohort"  # New axis
    timesteps_number = "timesteps_number"
    min_timestep = "min_timestep"
    max_timestep = "max_timestep"
    mean_timestep = "mean_timestep"
    # Forcing
    timestep = "timestep"
    resolution_latitude = "resolution_latitude"
    resolution_longitude = "resolution_longitude"
    initial_condition_production = "initial_condition_production"
    initial_condition_biomass = "initial_condition_biomass"
    # Kernel
    angle_horizon_sun = "angle_horizon_sun"
    compute_preproduction = "compute_preproduction"
    compute_initial_conditions = "compute_initial_conditions"


class ForcingLabels(StrEnum):
    """A single place to store all labels as declared in forcing module."""

    global_mask = "mask"
    mask_by_fgroup = "mask_fgroup"
    day_length = "day_length"
    avg_temperature_by_fgroup = "average_temperature"
    avg_acidity_by_fgroup = "average_acidity"
    primary_production_by_fgroup = "primary_production_by_fgroup"
    min_temperature = "min_temperature"
    mask_temperature = "mask_temperature"
    cell_area = "cell_area"
    mortality_field = "mortality_field"
    recruited = "recruited"
    preproduction = "preproduction"
    biomass = "biomass"
    temperature = "temperature"
    primary_production = "primary_production"
    acidity = "acidity"

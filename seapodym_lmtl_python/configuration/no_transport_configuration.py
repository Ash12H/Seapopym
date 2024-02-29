"""
This module contains the configuration of the model as a xarray.Dataset. It allows to store the model parameters and
the forcings (lazily).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import IO, Callable

import attrs
import cf_xarray  # noqa: F401
import xarray as xr

from seapodym_lmtl_python.configuration.base_configuration import BaseConfiguration
from seapodym_lmtl_python.configuration.no_transport_parameters import (
    FunctionalGroupUnit,
    FunctionalGroupUnitMigratoryParameters,
    FunctionalGroupUnitRelationParameters,
    NoTransportParameters,
    PathParameters,
)


class NoTransportLabels(StrEnum):
    """
    A single place to store all labels as :
    - used in the NoTransportConfiguration class
    - declared in no_transport_parameters
    module.
    """

    # Functional group
    fgroup_name = "name"
    energy_transfert = "energy_transfert"
    inv_lambda_max = "inv_lambda_max"
    inv_lambda_rate = "inv_lambda_rate"
    temperature_recruitment_max = "temperature_recruitment_max"
    temperature_recruitment_rate = "temperature_recruitment_rate"
    day_layer = "day_layer"
    night_layer = "night_layer"
    # Files
    fgroup = "functional_group"  # Equivalent to name


class NoTransportConfiguration(BaseConfiguration):
    """Configuration for the NoTransportModel."""

    def __init__(self: NoTransportConfiguration, parameters: NoTransportParameters) -> None:
        """Create a NoTransportConfiguration object."""
        self._parameters = parameters

    @property
    def parameters(self: NoTransportConfiguration) -> NoTransportParameters:
        """The attrs dataclass that stores all the model parameters."""
        return self._parameters

    @classmethod
    def parse(cls: NoTransportConfiguration, configuration_file: str | Path | IO) -> NoTransportConfiguration:
        """Parse the configuration file and create a NoTransportConfiguration object."""
        msg = "This method is not implemented yet."
        raise NotImplementedError(msg)

    def _load_forcings(self: NoTransportConfiguration, path_parameters: PathParameters, **kargs: dict) -> xr.Dataset:
        """Return the forcings as a xarray.Dataset."""
        if kargs is None:
            kargs = {}

        all_forcing = {}

        for forcing_name, forcing_value in attrs.asdict(path_parameters).items():
            if forcing_value is not None:
                all_forcing[forcing_name] = xr.open_dataset(
                    forcing_value["forcing_path"],
                    **kargs,
                )[forcing_value["name"]]

        return xr.Dataset(all_forcing)

    def _build_fgroup_dataset__generate_param_coords(
        self: NoTransportConfiguration,
        parameters: NoTransportParameters,
    ) -> tuple[dict, xr.DataArray]:
        """
        Helper method used by _build_fgroup_dataset to generate both the functional groups parameters as dictionary and
        the functional groups coordinates as xarray.DataArray.
        """

        def _rec_parameters(data: list) -> dict:
            """Recursivaly list all parameters."""
            all_param = {}
            for key in data[0]:
                if not isinstance(data[0][key], dict):
                    all_param[key] = [dic[key] for dic in data]
                else:
                    all_param.update(_rec_parameters([dic[key] for dic in data]))
            return all_param

        # 1. Generate a dictionary with all parameters as list
        grps: list[FunctionalGroupUnit] = parameters.functional_groups_parameters.functional_groups
        grps_param = _rec_parameters([attrs.asdict(grp) for grp in grps])

        # 2. Generate the coordinates (i.e. functional groups)
        f_group_coord_data = list(range(len(grps_param["name"])))
        f_group_coord = xr.DataArray(
            coords=(f_group_coord_data,),
            dims=(NoTransportLabels.fgroup,),
            name=NoTransportLabels.fgroup,
            attrs={  # cf_xarray convention
                "flag_values": f_group_coord_data,
                "flag_meanings": " ".join(grps_param["name"]),
                "standard_name": "functional group",
            },
            data=f_group_coord_data,
        )

        return grps_param, f_group_coord

    def _build_fgroup_dataset__generate_variables(
        self: NoTransportConfiguration,
        params: dict,
        classes_and_names: list[tuple[attrs.Attribute, str]],
    ) -> dict[str, tuple]:
        """
        Generate a dictionary where each key is a variable name and each value is a tuple of parameters. It can be used
        to create a xr.Dataset.
        """

        def _sel_attrs_meta(param_class: attrs.Attribute, attribut: str) -> dict:
            """Extract metadata from a specific attribut in an Attrs dataclass."""
            return next(filter(lambda x: x.name == attribut, attrs.fields(param_class))).metadata

        def _generate_tuple(param_class: attrs.Attribute, name: str) -> tuple:
            return ((NoTransportLabels.fgroup,), params[name], _sel_attrs_meta(param_class, name))

        return {name: _generate_tuple(cls, name) for cls, name in classes_and_names}

    def _build_fgroup_dataset(self: NoTransportConfiguration, parameters: NoTransportParameters) -> xr.Dataset:
        """
        Return the functional groups parameters as a xarray.Dataset.
        This function is used by the _build_model_configuration function.
        """
        (param_as_dict, f_group_coord) = self._build_fgroup_dataset__generate_param_coords(parameters)
        names_classes = [
            (FunctionalGroupUnit, NoTransportLabels.fgroup_name),
            (FunctionalGroupUnit, NoTransportLabels.energy_transfert),
            (FunctionalGroupUnitRelationParameters, NoTransportLabels.inv_lambda_max),
            (FunctionalGroupUnitRelationParameters, NoTransportLabels.inv_lambda_rate),
            (FunctionalGroupUnitRelationParameters, NoTransportLabels.temperature_recruitment_max),
            (FunctionalGroupUnitRelationParameters, NoTransportLabels.temperature_recruitment_rate),
            (FunctionalGroupUnitMigratoryParameters, NoTransportLabels.day_layer),
            (FunctionalGroupUnitMigratoryParameters, NoTransportLabels.night_layer),
        ]
        param_variables = self._build_fgroup_dataset__generate_variables(param_as_dict, names_classes)
        return xr.Dataset(param_variables, coords={NoTransportLabels.fgroup: f_group_coord})

    # TODO(Jules): Add the validation process
    # https://github.com/users/Ash12H/projects/3?pane=issue&itemId=54978078

    def as_dataset(self: NoTransportConfiguration) -> xr.Dataset:
        """Return the configuration as a xarray.Dataset."""
        fgroup = self._build_fgroup_dataset(self.parameters)
        forcings = self._load_forcings(self.parameters.path_parameters)
        return xr.merge([fgroup, forcings])

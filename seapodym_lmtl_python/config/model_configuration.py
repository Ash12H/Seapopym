"""
This module contains the configuration of the model as a xarray.Dataset. It allows to store the model parameters and
the forcings (lazily).
"""

import xarray as xr

from seapodym_lmtl_python.config import parameters


def _load_forcings(path_parameters: parameters.PathParameters) -> xr.Dataset:
    """Return the forcings as a xarray.Dataset."""
    pass


def _validate_forcings(parameters: parameters, forcings: xr.Dataset) -> None:
    """Validate the forcings against the parameters."""
    pass


def _build_model_configuration(
    parameters: parameters, forcings: xr.Dataset
) -> xr.Dataset:
    """Return the model configuration as a xarray.Dataset."""
    pass


def get_parameters(param: parameters) -> xr.Dataset:
    """
    Return the model configuration as a xarray.Dataset. Also run some test to ensure the parameters are consistant
    with the forcings.
    """
    forcings = _load_forcings(param.path_parameters)
    _validate_forcings(param, forcings)
    return _build_model_configuration(param, forcings)

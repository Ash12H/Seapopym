"""Defines type aliases for Seapopym standard types."""

import xarray as xr

from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels

type SeapopymState = xr.Dataset
"""SeapopymState is the model state that stores all forcings (IN/OUT) and parameters."""

type SeapopymForcing = xr.DataArray
"""SeapopymForcing is the forcing used as a xarray.DataArray in the SeapopymState."""

type ForcingName = ConfigurationLabels | ForcingLabels | str
"""ForcingName is the name of the forcing used as key in the SeapopymState."""

type SeapopymDims = CoordinatesLabels | str
"""SeapopymDims is the name of the dimensions used in the SeapopymState."""

type ForcingAttrs = dict[str, object]
"""ForcingAttrs is the attributes of the forcing used as xarray.DataArray.attrs in the SeapopymState."""

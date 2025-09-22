"""Store all default attributs for the seapopym xarray.DataArray."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seapopym.standard.labels import CoordinatesLabels
from seapopym.standard.units import StandardUnitsLabels, StandardUnitsRegistry

if TYPE_CHECKING:
    from collections.abc import Iterable


# TODO(Jules): Do the same for cohort axis
def functional_group_desc(f_group_coord_data: Iterable, groups_name: list[str]) -> dict:
    """Functional group attributs. Standard name is used as an accessor by cf_xarray."""
    return {
        "flag_values": str(f_group_coord_data),
        "flag_meanings": " ".join(groups_name),
        "standard_name": CoordinatesLabels.functional_group,
        "long_name": "functional group",
    }


"""dict: Functional group attributs."""

global_mask_desc = {
    "standard_name": "mask",
    "long_name": "mask",
    "flag_values": "[0, 1]",
    "flag_meanings": "0:land, 1:ocean",
}
"""dict: Global mask attributs."""

mask_by_fgroup_desc = global_mask_desc.copy()
"""dict: Mask by fgroup attributs."""


day_length_desc = {
    "long_name": "Day length",
    "standard_name": "day_length",
    "description": "Day length at the surface using Forsythe's method.",
    "units": StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.time),
}


average_temperature_by_fgroup_desc = {
    "long_name": "average sea temperature by fonctional group",
    "standard_name": "sea water temperature",
    "description": ("Average temperature by functional group according to their layer position during day and night."),
    "units": StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.temperature),
}

apply_coefficient_to_primary_production_desc = {
    "standard_name": "primary production",
    "long_name": "primary production by functional group",
    "description": "Primary production by functional group according to their energy transfert coefficient.",
    "units": StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.production),
}
"""dict: Apply coefficient to primary production attributs."""

min_temperature_by_cohort_desc = {
    "standard_name": "minimum temperature",
    "long_name": "minimum temperature by cohort",
    "description": "Minimum temperature to recruit a cohort according to its age.",
    "units": StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.temperature),
}
"""dict: Minimum temperature by cohort attributs."""

mask_temperature_desc = {
    "standard_name": "mask",
    "long_name": "cohort recruitment mask by functional group",
    "description": "Mask to recruit a cohort according to the temperature.",
    "flag_values": "[0, 1]",
    "flag_meanings": "0:not recruited, 1:recruited",
}
"""dict: Mask temperature attributs."""

compute_cell_area_desc = {
    "standard_name": "cell_area",
    "long_name": "cell area",
    "description": "Cell area computed from the latitude and longitude centroid.",
    "units": f"{StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.height)}**2",
}
"""dict: Compute cell area attributs."""

mortality_field_desc = {
    "standard_name": "mortality",
    "long_name": "mortality coefficient",
    "description": "Mortality coefficient according to the temperature.",
}
"""dict: Mortality field attributs."""

recruited_desc = {
    "standard_name": "production",
    "long_name": "production",
    "units": StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.production),
}

preproduction_desc = {
    "standard_name": "preproduction",
    "long_name": "pre-production",
    "description": "The entire population before recruitment, divided into cohorts.",
    "units": StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.production),
}

biomass_desc = {
    "long_name": "biomass",
    "units": StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.biomass),
    "description": "The biomass of the recruited individuals.",
}
average_acidity_by_fgroup_desc = {
    "long_name": "average acidity (pH) by fonctional group",
    "standard_name": "sea water acidity (pH)",
    "description": ("Average acidity (pH) by functional group according to their layer position during day and night."),
    "units": StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.acidity),
}
mortality_acidity_field_desc = {
    "standard_name": "mortality",
    "long_name": "mortality coefficient (T, pH)",
    "description": "Mortality coefficient according to the temperature and acidity (pH).",
}

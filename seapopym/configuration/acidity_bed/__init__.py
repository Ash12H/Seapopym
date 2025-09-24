"""Configuration for acidity model with Bednarsek mortality equation."""

from seapopym.configuration.acidity_bed.configuration import AcidityBedConfiguration
from seapopym.configuration.acidity_bed.functional_group_parameter import (
    FunctionalGroupParameter,
    FunctionalGroupUnit,
    FunctionalTypeParameter,
)

__all__ = [
    "AcidityBedConfiguration",
    "FunctionalGroupParameter",
    "FunctionalGroupUnit",
    "FunctionalTypeParameter",
]
"""Seapopym standard types, protocols and utilities.

Central module providing:
- Type definitions and protocols
- Coordinate management and validation
- Units and attributes handling
- CF-compliant data structures
"""

from seapopym.standard.coordinate_authority import CoordinateAuthority, coordinate_authority
from seapopym.standard.labels import CoordinatesLabels, ConfigurationLabels, ForcingLabels
from seapopym.standard.protocols import (
    ConfigurationProtocol,
    ModelProtocol,
    TemplateProtocol,
    ChunkParameterProtocol,
    KernelParameterProtocol,
    ForcingParameterProtocol,
    FunctionalGroupParameterProtocol,
)
from seapopym.standard.types import SeapopymState, SeapopymForcing, ForcingName, SeapopymDims

__all__ = [
    # Core types
    "SeapopymState",
    "SeapopymForcing",
    "ForcingName",
    "SeapopymDims",
    # Labels
    "CoordinatesLabels",
    "ForcingLabels",
    "ConfigurationLabels",
    # Protocols
    "ConfigurationProtocol",
    "ModelProtocol",
    "TemplateProtocol",
    "ChunkParameterProtocol",
    "KernelParameterProtocol",
    "ForcingParameterProtocol",
    "FunctionalGroupParameterProtocol",
    # Coordinate authority
    "CoordinateAuthority",
    "coordinate_authority",
]
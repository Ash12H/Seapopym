"""Class to store acidity model configuration"""

from __future__ import annotations

from attrs import field, frozen, validators
from seapopym.configuration.no_transport.parameter import ForcingParameters,NoTransportParameters
from seapopym.configuration.parameters.parameter_forcing import ForcingUnit
from seapopym.standard.units import StandardUnitsLabels
from seapopym.configuration.no_transport.configuration import NoTransportConfiguration



@frozen(kw_only=True)
class AcidityForcingParameters(ForcingParameters):
    """
    This data class extends ForcingParameters to include an acidity forcing field.
    """
    
    acidity: ForcingUnit = field(
        validator=validators.instance_of(ForcingUnit),
        metadata={"description": "Path to the acidity field."},
    )

    def _check_units(self:AcidityForcingParameters) -> ForcingUnit:
        super()._check_units()
        self.acidity.with_units(StandardUnitsLabels.acidity.units, in_place=True)

    def __attrs_post_init__(self:AcidityForcingParameters) -> None:
        forcings = [
            self.temperature,
            self.primary_production,
            self.mask,
            self.day_length,
            self.cell_area,
            self.initial_condition_production,
            self.initial_condition_biomass,
            self.acidity,
        ]
        forcings = [field for field in forcings if field is not None]
        self._set_timestep(forcings)
        self._set_resolution(forcings)
        self._check_units()


@frozen(kw_only=True)
class AcidityParameters(NoTransportParameters):
    """Adding the acidity Forcage to the main data class"""
    forcing_parameters: AcidityForcingParameters = field(
        metadata={"description": "All the paths to the forcings."}
    )


class AcidityConfiguration(NoTransportConfiguration):
    """Configuration for the acidity 1D model""" 

    def __init__(self: NoTransportConfiguration, parameters: AcidityParameters) -> None:
        """Create an AcidityConfiguration object."""
        self._parameters = parameters

# question : est-ce que as_dataset tolerera d'avoir acidity en forcing parameters ???

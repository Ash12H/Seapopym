from __future__ import annotations

from functools import partial

from attrs import field, frozen, validators

from seapopym.configuration import no_transport
from seapopym.configuration.validation import verify_forcing_init
from seapopym.standard.labels import ForcingLabels
from seapopym.standard.units import StandardUnitsLabels


@frozen(kw_only=True)
class ForcingParameter(no_transport.ForcingParameter):
    """This data class extends ForcingParameters to include an acidity forcing field."""

    acidity: no_transport.ForcingUnit = field(
        alias=ForcingLabels.acidity,
        converter=partial(
            verify_forcing_init, unit=StandardUnitsLabels.acidity.units, parameter_name=ForcingLabels.acidity
        ),
        validator=validators.instance_of(no_transport.ForcingUnit),
        metadata={"description": "Path to the acidity field."},
    )

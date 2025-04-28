from __future__ import annotations

from attrs import field, frozen, validators

from seapopym.configuration import no_transport


@frozen(kw_only=True)
class ForcingParameter(no_transport.ForcingParameter):
    """This data class extends ForcingParameters to include an acidity forcing field."""

    acidity: no_transport.ForcingUnit = field(
        validator=validators.instance_of(no_transport.ForcingUnit),
        metadata={"description": "Path to the acidity field."},
    )

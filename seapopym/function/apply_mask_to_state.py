"""
This module provides a function to apply a mask to a Seapopym state dataset.

The `apply_mask_to_state` function checks if a global mask is present in the state dataset and applies it to filter the
dataset accordingly. If no mask is found, the state dataset is returned unchanged.
"""

from seapopym.standard.labels import ForcingLabels
from seapopym.standard.types import SeapopymState


def apply_mask_to_state(state: SeapopymState) -> SeapopymState:
    """Apply a mask to a state dataset."""
    if ForcingLabels.global_mask in state:
        return state.where(state[ForcingLabels.global_mask])
    return state

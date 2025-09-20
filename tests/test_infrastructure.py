"""Test to verify that the test infrastructure is working correctly."""

import pytest
import numpy as np
import xarray as xr

from seapopym.configuration.no_transport.forcing_parameter import ChunkParameter


def test_infrastructure_basic():
    """Basic test to verify pytest is working."""
    assert True


def test_fixtures_work(sample_coordinates, sample_chunk_parameter):
    """Test that fixtures are properly loaded."""
    # Test coordinates fixture
    assert 'time' in sample_coordinates
    assert 'latitude' in sample_coordinates
    assert 'longitude' in sample_coordinates

    # Test that coordinates have standardized names
    assert sample_coordinates['time'].name == 'T'
    assert sample_coordinates['latitude'].name == 'Y'
    assert sample_coordinates['longitude'].name == 'X'

    # Test chunk parameter fixture
    assert isinstance(sample_chunk_parameter, ChunkParameter)
    assert sample_chunk_parameter.Y == 18
    assert sample_chunk_parameter.X == 36


def test_import_seapopym():
    """Test that seapopym can be imported."""
    import seapopym
    import seapopym.configuration.no_transport.forcing_parameter
    import seapopym.standard.coordinates

    assert seapopym is not None


@pytest.mark.coordinates
def test_marker_example():
    """Example test with custom marker."""
    assert True
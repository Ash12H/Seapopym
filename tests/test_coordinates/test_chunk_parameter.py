"""Tests for ChunkParameter coordinate standardization."""

import pytest
from seapopym.configuration.no_transport.forcing_parameter import ChunkParameter


@pytest.mark.coordinates
class TestChunkParameter:
    """Test class for ChunkParameter functionality."""

    def test_as_dict_returns_standardized_coordinates(self):
        """Test that as_dict() returns T/Y/X/Z coordinate names."""
        chunk = ChunkParameter(Y=180, X=360)
        result = chunk.as_dict()

        expected = {
            'functional_group': 1,  # default value
            'Y': 180,
            'X': 360,
            'T': -1  # default value for time
        }

        assert result == expected

    def test_as_dict_with_custom_functional_group(self):
        """Test as_dict() with custom functional_group value."""
        chunk = ChunkParameter(functional_group=5, Y=90, X=180)
        result = chunk.as_dict()

        expected = {
            'functional_group': 5,
            'Y': 90,
            'X': 180,
            'T': -1
        }

        assert result == expected

    def test_as_dict_with_none_values(self):
        """Test as_dict() when Y or X are None."""
        chunk = ChunkParameter(Y=None, X=180)
        result = chunk.as_dict()

        expected = {
            'functional_group': 1,
            'X': 180,
            'T': -1
        }

        assert result == expected
        assert 'Y' not in result  # None values should be excluded

    def test_as_dict_excludes_old_coordinate_names(self):
        """Test that as_dict() doesn't contain old coordinate names."""
        chunk = ChunkParameter(Y=180, X=360)
        result = chunk.as_dict()

        # Ensure old names are not present
        assert 'latitude' not in result
        assert 'longitude' not in result
        assert 'time' not in result

    def test_chunk_parameter_defaults(self):
        """Test ChunkParameter default values."""
        chunk = ChunkParameter()
        result = chunk.as_dict()

        expected = {
            'functional_group': 1,
            'T': -1
        }

        assert result == expected
        # Y and X should not be present when None
        assert 'Y' not in result
        assert 'X' not in result

    def test_fixture_integration(self, sample_chunk_parameter, sample_chunk_dict):
        """Test integration with conftest.py fixtures."""
        # Test that fixture works correctly
        assert isinstance(sample_chunk_parameter, ChunkParameter)

        result = sample_chunk_parameter.as_dict()
        assert result == sample_chunk_dict

    def test_time_coordinate_always_present(self):
        """Test that T coordinate is always present in as_dict()."""
        chunk = ChunkParameter(Y=None, X=None)
        result = chunk.as_dict()

        assert 'T' in result
        assert result['T'] == -1

    @pytest.mark.parametrize("y_value,x_value", [
        (10, 20),
        (180, 360),
        (1, 1),
        (None, 50),
        (50, None)
    ])
    def test_as_dict_parametrized(self, y_value, x_value):
        """Test as_dict() with various Y/X combinations."""
        chunk = ChunkParameter(Y=y_value, X=x_value)
        result = chunk.as_dict()

        # T and functional_group should always be present
        assert 'T' in result
        assert 'functional_group' in result

        # Y and X should be present only if not None
        if y_value is not None:
            assert result['Y'] == y_value
        else:
            assert 'Y' not in result

        if x_value is not None:
            assert result['X'] == x_value
        else:
            assert 'X' not in result
"""Tests for Phase 1 protocol compatibility.

Verifies that existing classes automatically implement the new protocols
without requiring code changes.
"""

from typing import Any

import pytest

from seapopym.configuration.no_transport.forcing_parameter import ChunkParameter, ForcingUnit
from seapopym.configuration.no_transport.functional_group_parameter import FunctionalGroupUnit
from seapopym.configuration.no_transport.kernel_parameter import KernelParameter
from seapopym.standard.protocols import (
    ChunkParameterProtocol,
    ForcingUnitProtocol,
    FunctionalGroupUnitProtocol,
    KernelParameterProtocol,
)


@pytest.mark.protocols
class TestPhase1ProtocolCompatibility:
    """Test that existing classes implement Phase 1 protocols."""

    def test_chunk_parameter_implements_protocol(self):
        """Test that ChunkParameter implements ChunkParameterProtocol."""
        chunk = ChunkParameter(Y=180, X=360)

        # Test that it has the required method
        assert hasattr(chunk, 'as_dict')
        assert callable(chunk.as_dict)

        # Test that the method works correctly
        result = chunk.as_dict()
        assert isinstance(result, dict)
        assert 'Y' in result
        assert 'X' in result

        # Type checking should pass (verified by mypy)
        def accepts_chunk_protocol(param: ChunkParameterProtocol) -> dict:
            return param.as_dict()

        # This should work without errors
        result = accepts_chunk_protocol(chunk)
        assert isinstance(result, dict)

    def test_kernel_parameter_implements_protocol(self):
        """Test that KernelParameter implements KernelParameterProtocol."""
        kernel = KernelParameter()

        # Since KernelParameterProtocol has no required methods,
        # any object should implement it
        def accepts_kernel_protocol(param: KernelParameterProtocol) -> None:
            # Just verify we can accept the object
            assert param is not None

        # This should work without errors
        accepts_kernel_protocol(kernel)

    def test_forcing_unit_implements_protocol(self, sample_dataarray_standardized):
        """Test that ForcingUnit implements ForcingUnitProtocol."""
        forcing_unit = ForcingUnit(forcing=sample_dataarray_standardized)

        # Test that it has the required attribute
        assert hasattr(forcing_unit, 'forcing')
        assert forcing_unit.forcing is not None

        # Type checking should pass
        def accepts_forcing_unit_protocol(unit: ForcingUnitProtocol) -> Any:
            return unit.forcing

        # This should work without errors
        result = accepts_forcing_unit_protocol(forcing_unit)
        assert result is not None

    def test_functional_group_unit_implements_protocol(self):
        """Test that FunctionalGroupUnit implements FunctionalGroupUnitProtocol."""
        # Create a minimal functional group unit for testing
        # (using mock data since full creation requires complex setup)

        # Test that the class has the required attributes and methods
        assert hasattr(FunctionalGroupUnit, 'name')
        assert hasattr(FunctionalGroupUnit, 'migratory_type')
        assert hasattr(FunctionalGroupUnit, 'functional_type')
        assert hasattr(FunctionalGroupUnit, 'to_dataset')
        assert callable(FunctionalGroupUnit.to_dataset)

    def test_protocol_duck_typing(self):
        """Test that protocols work with duck typing (any compatible object)."""

        # Create a simple class that implements ChunkParameterProtocol
        class MockChunkParameter:
            def as_dict(self) -> dict:
                return {'T': -1, 'Y': 10, 'X': 20}

        mock_chunk = MockChunkParameter()

        # This should work without inheritance
        def accepts_chunk_protocol(param: ChunkParameterProtocol) -> dict:
            return param.as_dict()

        result = accepts_chunk_protocol(mock_chunk)
        assert result == {'T': -1, 'Y': 10, 'X': 20}

    def test_protocol_type_checking_helper(self):
        """Helper test to verify type annotations work correctly."""
        # This test primarily exists for mypy verification

        def process_chunk(chunk: ChunkParameterProtocol) -> dict:
            """Function that accepts any object implementing ChunkParameterProtocol."""
            return chunk.as_dict()

        def process_kernel(kernel: KernelParameterProtocol) -> None:
            """Function that accepts any object implementing KernelParameterProtocol."""
            pass

        def process_forcing_unit(unit: ForcingUnitProtocol) -> Any:
            """Function that accepts any object implementing ForcingUnitProtocol."""
            return unit.forcing

        def process_functional_group(group: FunctionalGroupUnitProtocol) -> str:
            """Function that accepts any object implementing FunctionalGroupUnitProtocol."""
            return group.name

        # These functions should accept our existing classes
        # (actual tests are in the individual test methods above)
        assert callable(process_chunk)
        assert callable(process_kernel)
        assert callable(process_forcing_unit)
        assert callable(process_functional_group)
"""Tests for Phase 6 template Protocol migration.

Verifies that BaseTemplate has been successfully migrated from ABC to Protocol.
"""

import abc
import pytest

from seapopym.core.template import BaseTemplate, TemplateUnit, Template
from seapopym.standard.protocols import TemplateProtocol


@pytest.mark.protocols
class TestPhase6TemplateMigration:
    """Test that template classes have been migrated from ABC to Protocol."""

    def test_base_template_no_longer_inherits_abc(self):
        """Test that BaseTemplate no longer inherits from abc.ABC."""

        # BaseTemplate should not directly inherit from abc.ABC
        assert abc.ABC not in BaseTemplate.__bases__

        # BaseTemplate should not have abc.ABC in its MRO
        assert abc.ABC not in BaseTemplate.__mro__

        # BaseTemplate should not be an instance of ABCMeta
        assert not isinstance(BaseTemplate, abc.ABCMeta)

        # BaseTemplate should be a regular class
        assert isinstance(BaseTemplate, type)

    def test_base_template_implements_template_protocol(self):
        """Test that BaseTemplate implements TemplateProtocol via duck typing."""

        def accepts_template_protocol(template_class: type[TemplateProtocol]) -> bool:
            """Function that accepts TemplateProtocol."""
            return hasattr(template_class, 'generate') and callable(template_class.generate)

        # This should work without ABC inheritance
        assert accepts_template_protocol(BaseTemplate)

    def test_base_template_generate_raises_not_implemented(self):
        """Test that BaseTemplate's generate method raises NotImplementedError."""

        # Creating BaseTemplate directly should be possible (no ABC preventing it)
        template = BaseTemplate()

        # generate should raise NotImplementedError
        mock_state = {"test": "state"}
        with pytest.raises(NotImplementedError, match="must implement generate"):
            template.generate(mock_state)

    def test_template_unit_implements_protocol(self):
        """Test that TemplateUnit implements TemplateProtocol."""

        # TemplateUnit should have generate method
        assert hasattr(TemplateUnit, 'generate')
        assert callable(TemplateUnit.generate)

        # Check inheritance from BaseTemplate
        assert issubclass(TemplateUnit, BaseTemplate)

    def test_template_implements_protocol(self):
        """Test that Template implements TemplateProtocol."""

        # Template should have generate method
        assert hasattr(Template, 'generate')
        assert callable(Template.generate)

        # Check inheritance from BaseTemplate
        assert issubclass(Template, BaseTemplate)

    def test_template_protocol_duck_typing(self):
        """Test that TemplateProtocol works with duck typing."""

        class MockTemplate:
            def generate(self, state):
                return {"mock": "template"}

        mock_template = MockTemplate()

        def use_template_protocol(template: TemplateProtocol):
            """Function that uses template protocol."""
            return template.generate({"test": "state"})

        result = use_template_protocol(mock_template)
        assert result == {"mock": "template"}

    def test_base_template_can_be_instantiated_directly(self):
        """Test that BaseTemplate can be instantiated directly (not abstract anymore)."""

        # This should work without ABC restrictions
        template = BaseTemplate()
        assert template is not None
        assert isinstance(template, BaseTemplate)

    def test_migration_maintains_protocol_compatibility(self):
        """Test that migration maintains protocol compatibility."""

        def process_template(template_class: type[TemplateProtocol]) -> str:
            """Function that processes a template class implementing TemplateProtocol."""
            assert hasattr(template_class, 'generate')
            assert callable(template_class.generate)
            return "Protocol compatible"

        # All template classes should be compatible
        assert process_template(BaseTemplate) == "Protocol compatible"
        assert process_template(TemplateUnit) == "Protocol compatible"
        assert process_template(Template) == "Protocol compatible"

    def test_template_hierarchy_maintains_functionality(self):
        """Test that template class hierarchy maintains functionality."""

        def check_template_hierarchy(template_class: type[TemplateProtocol]) -> dict:
            """Function that checks template hierarchy."""
            result = {}
            result['has_generate'] = hasattr(template_class, 'generate')
            result['is_callable'] = callable(getattr(template_class, 'generate', None))
            result['inherits_base_template'] = issubclass(template_class, BaseTemplate)
            return result

        # Check BaseTemplate
        base_result = check_template_hierarchy(BaseTemplate)
        assert base_result['has_generate']
        assert base_result['is_callable']
        assert base_result['inherits_base_template']

        # Check TemplateUnit
        unit_result = check_template_hierarchy(TemplateUnit)
        assert unit_result['has_generate']
        assert unit_result['is_callable']
        assert unit_result['inherits_base_template']

        # Check Template
        template_result = check_template_hierarchy(Template)
        assert template_result['has_generate']
        assert template_result['is_callable']
        assert template_result['inherits_base_template']
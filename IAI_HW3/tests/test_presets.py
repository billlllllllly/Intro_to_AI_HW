import pytest

from main import PRESETS


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_have_three_distinct_members(preset):
    members = PRESETS[preset]
    assert len(members) == 3
    assert len({member.name for member in members}) == 3
    assert len({member.seed for member in members}) == 3

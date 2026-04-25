"""Palace test configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_MESH_COUNT_KEYS = ("elements", "nodes", "tetrahedra")
_MESH_COUNT_RTOL = 0.01


class MeshRegressionFixture:
    """Single-YAML regression fixture: exact groups, 1% tolerance on counts."""

    def __init__(self, original_datadir: Path, request: pytest.FixtureRequest) -> None:
        """Store the reference directory and pytest request."""
        self._original_datadir = original_datadir
        self._request = request

    def check(self, snapshot: dict) -> None:
        """Assert snapshot matches reference; regenerate with --force-regen."""
        force_regen = self._request.config.getoption("force_regen", default=False)
        ref_path = self._original_datadir / f"{self._request.node.name}.yml"

        if force_regen or not ref_path.exists():
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref_path.write_text(yaml.dump(snapshot, default_flow_style=False))
            pytest.fail(
                f"Reference file written: {ref_path}. Re-run without --force-regen."
            )

        ref = yaml.safe_load(ref_path.read_text())

        assert snapshot["groups"] == ref["groups"], (
            "Mesh group names do not match reference."
        )

        assert snapshot["mesh"]["invalid_elements"] == 0, "Mesh has invalid elements."
        for key in _MESH_COUNT_KEYS:
            obtained = snapshot["mesh"][key]
            expected = ref["mesh"][key]
            assert obtained == pytest.approx(expected, rel=_MESH_COUNT_RTOL), (
                f"Mesh count '{key}' = {obtained} differs from reference {expected} "
                f"by more than {_MESH_COUNT_RTOL:.0%}."
            )


@pytest.fixture
def mesh_regression(original_datadir, request):
    """Fixture for mesh snapshot regression with 1% tolerance on element counts."""
    return MeshRegressionFixture(original_datadir, request)

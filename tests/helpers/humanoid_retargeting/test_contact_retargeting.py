"""Interaction-retargeting import tests (full solves are covered by the golden test)."""


class TestPresets:
    def test_import_loads_no_preset_module(self):
        # Importing the shared API must not import robot presets or torch.
        import subprocess
        import sys

        code = (
            "import sys, robokit.helpers.humanoid_retargeting; "
            "assert 'torch' not in sys.modules; "
            "assert not any(m.startswith('robokit.helpers.humanoid_retargeting.presets.') for m in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr

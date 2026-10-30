"""Test numpy+warp `robokit.smplx` imports without the optional `pt-*` torch dependency.

Isolation steps:
1. Start a subprocess so prior imports cannot hide a dependency.
2. Set `sys.modules["torch"] = None` to make any torch import raise ImportError.
3. Import the runtime and its loaders.
"""

import subprocess
import sys


_IMPORTS = """
import sys
sys.modules["torch"] = None
import robokit.smplx
from robokit.helpers.hand_retargeting.dexycb_utils import compute_world_hand_geometry
from robokit.helpers.humanoid_retargeting.loaders import SmplxMotion, get_smplx_motion, load_smplx_file
"""


class TestNoTorchRuntime:
    def test_smplx_runtime_imports_without_torch(self):
        subprocess.run([sys.executable, "-c", _IMPORTS], check=True)

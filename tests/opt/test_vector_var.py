from enum import Enum

import numpy as np
import warp as wp

from robokit.opt import VarValues, VectorVar


class Key(Enum):
    X = 0


class TestVectorVar:
    def test_integrate_clone_gather_accept(self):
        x = wp.array([[1.0, 2.0], [3.0, 4.0]], dtype=wp.float32)
        var = VectorVar(x)
        clone = var.clone()
        velocity = wp.array([[0.5, 1.0], [1.5, 2.0]], dtype=wp.float32)
        mask = wp.array([1.0, 0.0], dtype=wp.float32)

        out = VectorVar(wp.empty_like(x))
        assert var.integrate(velocity, out=out, tangent_mask=mask, weight_decay=0.1) is out
        np.testing.assert_allclose(out.x.numpy(), [[1.4, 2.0], [4.2, 4.0]], atol=1e-6)
        np.testing.assert_array_equal(var.x.numpy(), [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(clone.x.numpy(), var.x.numpy())

        indices = wp.array([1, 0], dtype=wp.int32)
        gathered = var.gather(indices)
        np.testing.assert_array_equal(gathered.x.numpy(), [[3.0, 4.0], [1.0, 2.0]])

        proposed = VectorVar(wp.array([[9.0, 8.0], [7.0, 6.0]], dtype=wp.float32))
        assert var.accept(wp.array([0, 1], dtype=wp.int32), proposed) is var
        np.testing.assert_array_equal(var.x.numpy(), [[1.0, 2.0], [7.0, 6.0]])

    def test_var_values_wrapper(self):
        var = VectorVar(wp.zeros((3, 4), dtype=wp.float32))
        values = VarValues.from_var(var, Key.X)

        assert values.get(Key.X) is var
        assert values.batch_size == 3
        assert values.tangent_dim == 4

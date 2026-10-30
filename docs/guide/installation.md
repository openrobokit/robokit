# Installation

RoboKit needs Python 3.8 or newer; CI tests 3.8 through 3.14 at the oldest allowed and newest dependency versions.

=== "uv"
```shell
uv add robokit
```
=== "pip"
```shell
pip install robokit
```

## From source

=== "uv"
```shell
uv add git+https://github.com/openrobokit/robokit
```
=== "pip"
```shell
pip install git+https://github.com/openrobokit/robokit
```

To work on RoboKit itself, follow the [Development Guide](../dev.md#setup).

## Extras

| Extra | What it adds |
| --- | --- |
| `mj` | MJCF loading through `mujoco`; MJCF assembly needs mujoco 3.3.4 or newer (Python 3.9+). |
| `mjwarp` | MuJoCo Warp, for SimMPC's rollouts (Python 3.10+). |
| `pt-cu126`, `pt-cu118`, `pt-cu128`, `pt-cpu` | PyTorch, for the parts that need it; pick the one matching your CUDA toolkit. |

The mask-alignment term and camera calibration also need `nvdiffrast` built for your exact PyTorch and CUDA, such as
`pip install "nvdiffrast==0.3.3+pt2.11.0cu126" --extra-index-url https://jianglongye.github.io/torch_packages_builder`.

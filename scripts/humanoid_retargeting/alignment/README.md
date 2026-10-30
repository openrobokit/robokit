# Humanoid alignment

Fit a retargeting preset's `scale_table` + per-link `position_offset` so SMPL-X keypoints
land on the robot. All the machinery lives in
`robokit.helpers.humanoid_retargeting.alignment`; the fit is a closed-form least squares
(the target mapping is linear in the unknowns), so it solves instantly - the editor
re-aligns live on every slider edit.

## Workflow

```shell
# Interactive (any robot): review/fix pose pairs in the browser, watch the aligned
# errors update live, then "Save poses + preset".
uv run scripts/humanoid_retargeting/alignment/align.py --robot g1

# Headless: initialize missing pairs, solve, and save both Python presets.
uv run scripts/humanoid_retargeting/alignment/align.py --robot gr3 --headless

# Check the result against deployment: tracking error + foot float on real SFU clips.
uv run scripts/humanoid_retargeting/alignment/eval.py --robot g1
```

A new robot starts with an empty `presets/alignment/<robot>.py` dictionary. Missing robot
poses are initialized from a single-frame IK solve, then corrected in the editor.

## Editor

The header always shows the current pose, its mean residual, a quality bar, and whether
there are unsaved edits; everything else sits in four tabs.

| tab | what it is for |
|---|---|
| **Poses** | pick a pose (dropdown / Prev / Next), see every pose's before → after error, `⚠` on pairs whose limb directions disagree, and why |
| **Joints** | `IK init` one pose or all, `Mirror` one side onto the other (roll/yaw flip, sagittal joints keep sign), `Zero all joints`, then per-joint sliders foldered by body part (URDF limits enforced) |
| **View** | camera presets (Front / 3-4 / Side / Top, derived from the body's own facing), X-ray robot, and per-layer visibility |
| **Fit** | the worst-tracked links for this pose, plus the fitted `scale_table` and `human_height_assumption` |

Reading the 3D view: spheres are the targets the runtime would track, dots are the robot
links, and both they and the residual lines run green (matched) to red (60 mm+). A label
marks the worst link. Well-matched targets sit *inside* the robot meshes - turn on **X-ray
robot** to see them. Edits persist when you switch poses; "Save" writes
`presets/alignment/<robot>.py` and `presets/<robot>_aligned.py`.

## What makes a good pose pair

A pair must put BOTH bodies in the same physical configuration - the fit matches human
keypoints onto the robot's fixed FK, so a mismatched pair (human arms raised, robot arms
lowered) biases the learned scales/offsets. The editor's limb-direction audit flags
exactly this: unlike the keypoint residual, a direction error cannot be absorbed by
scales/offsets. Respect the robot's joint limits when posing (e.g. G1 shoulder_roll tops
out at 2.25, so lateral raises are specced to what the robot can reach).

Both bodies stand grounded on z=0 in the runtime Z-up frame (human: lowest mesh vertex;
robot: foot-sole geometry), so the learned mapping encodes the true robot-vs-human stature
ratio and the retargeted robot keeps its feet on the ground. The alignment body's T-pose
height is exported as `human_height_assumption`, which the runtime compares with each
motion's `human_heights` input.

See `src/robokit/examples/humanoid_retargeting/09_alignment.py` for a before/after motion demo.

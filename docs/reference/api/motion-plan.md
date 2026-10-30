# Motion Planning

::: robokit.helpers.motion_plan.MotionPlanner
::: robokit.helpers.motion_plan.MotionPlannerConfig
::: robokit.helpers.motion_plan.MotionPlanResult

## Objective configuration

`GradientTrajectoryOptimizerConfig` and `MppiTrajectoryOptimizerConfig` accept
`terms`, a list of `ResidualTask` templates. Objective weights and cost options
are task constructor parameters. Use `config.task(name)` to retrieve a named task;
for `TrajectoryTask`, configure its `dense_task`. `deepcopy(config)` creates an independent preset
whose task attributes can be assigned directly. Config-level cost-weight arguments have
been removed. IK objectives are supplied through `MotionPlannerConfig.goal_ik`.

# Project: SAC Curriculum Learning for Tendon-Driven Arm (MuJoCo)

## Introduction
This project trains a tendon-driven redundant robotic arm in MuJoCo using **Stable-Baselines3 SAC** with a **3-stage curriculum**:

- **Stage 1 (Basic reach / fixed target):** learn stable reaching behavior with a fixed goal.
- **Stage 2 (Generalization / random target):** randomize goals to improve generalization; optionally use **expert reward shaping**.
- **Stage 3 (Obstacle avoidance):** introduce obstacles and path-aware rewards; optionally enable a **temporal feature extractor** (e.g., LSTM/GRU/TCN) together with **frame stacking** to handle partial observability and improve avoidance stability.

Optional guidance is provided by **expert reward shaping**: the environment reward is augmented by an extra term based on the distance between the current state and the closest expert state.

---

## System Pseudocode

```text
INPUT:
  - stage ∈ {stage1, stage2, stage3, all}
  - TrainingConfig: per-stage settings (mjcf_path, timesteps, fixed_target, max_steps,
                   use_expert, expert_data_path, expert_ratio, reward_shaping_weight,
                   use_temporal, temporal_type, frame_stack, temporal_hparams)
  - SAC hyperparameters (lr, buffer_size, batch_size, gamma, tau, net_arch, ...)

MAIN():
  cfg = TrainingConfig()
  cfg = apply_cli_overrides(cfg)

  if stage == "all":
     for s in [stage1, stage2, stage3]:
        model_path = TRAIN_STAGE(s, input_model_path = last_stage_model_path)
        last_stage_model_path = model_path
  else:
     input_model_path = cli_input_model_path OR cfg.get_previous_stage_model_path(stage)
     TRAIN_STAGE(stage, input_model_path)


TRAIN_STAGE(stage, input_model_path):
  scfg = cfg.stage[stage]

  # 1) Load expert trajectories (optional)
  expert_trajs = []
  if scfg.use_expert == True:
     expert_trajs = LOAD_EXPERT_PKLS(scfg.expert_data_path, scfg.expert_ratio)

  # 2) Create environments
  # train env: may include expert reward shaping + optional frame stack
  train_env = MAKE_ENV(mjcf=scfg.mjcf_path,
                       fixed_target=scfg.fixed_target,
                       max_steps=scfg.max_steps,
                       obstacles=(stage==stage3),
                       expert_trajs=expert_trajs if scfg.reward_shaping_weight>0 else None,
                       expert_weight=scfg.reward_shaping_weight,
                       frame_stack=scfg.frame_stack if scfg.use_temporal else 1)

  # eval env: no expert shaping (to avoid metric contamination)
  eval_env  = MAKE_ENV(mjcf=scfg.mjcf_path,
                       fixed_target=scfg.fixed_target,
                       max_steps=scfg.max_steps,
                       obstacles=(stage==stage3),
                       expert_trajs=None,
                       expert_weight=0,
                       frame_stack=scfg.frame_stack if scfg.use_temporal else 1)

  # 3) Build SAC model (optional temporal feature extractor)
  policy_kwargs = DEFAULT_POLICY_KWARGS(net_arch=cfg.net_arch, activation=cfg.activation)
  if scfg.use_temporal:
     policy_kwargs.features_extractor_class  = TEMPORAL_EXTRACTOR(scfg.temporal_type)
     policy_kwargs.features_extractor_kwargs = scfg.temporal_hparams

  model = SAC(policy="MlpPolicy", env=train_env, policy_kwargs=policy_kwargs, **cfg.sac_hparams)

  # 4) Resume / transfer from previous stage when compatible
  if input_model_path exists AND scfg.use_temporal == False:
     old = LOAD_SAC(input_model_path)
     if OBS_SHAPE_MATCH(old.env, train_env):
        model.load_parameters_from(old)

  # 5) Train with callbacks (eval/checkpoint/progress)
  callbacks = [EvalCallback(eval_env), CheckpointCallback(...), ProgressCallback(...)]
  model.learn(total_timesteps=scfg.timesteps, callback=callbacks)

  # 6) Save final model
  save_path = SAVE(model, f"models/{stage}/sac_stage_final.zip")
  return save_path


LOAD_EXPERT_PKLS(path, ratio):
  pkl_files = list_all(path, "*.pkl")
  k = max(1, floor(len(pkl_files) * ratio)) if ratio>0 else 0
  return concat(pickle_load(pkl_files[0:k]))  # list of trajectories


MAKE_ENV(...):
  env = TendonArmEnv(mjcf_path, fixed_target, max_steps, obstacles=...)
  if expert_trajs is not None:
     env = ExpertRewardShapingWrapper(env, expert_trajs, expert_weight)
  if frame_stack > 1:
     env = FrameStackWrapper(env, n_stack=frame_stack)
  return env


from algos.misa import MISA
from utilities.utils import WandBLogger, define_flags_with_default

FLAGS_DEF = define_flags_with_default(
  algo="MISA",
  type="model-free",
  env="walker2d-medium-v2",
  dataset='d4rl',
  max_traj_length=1000,
  save_model=False,
  seed=42,
  batch_size=256,
  reward_scale=1,
  reward_bias=0,
  clip_action=0.999,
  encoder_arch="64-64",
  policy_arch="256-256",
  qf_arch="256-256",
  orthogonal_init=False,
  policy_log_std_multiplier=1.0,
  policy_log_std_offset=-1.0,
  algo_cfg=MISA.get_default_config(),
  n_epochs=1200,
  bc_epochs=0,
  n_train_step_per_epoch=1000,
  eval_period=10,
  eval_n_trajs=10,
  # configs for trining scheme
  logging=WandBLogger.get_default_config(),
  use_layer_norm=False,
  activation="elu",
  obs_norm=False,
)
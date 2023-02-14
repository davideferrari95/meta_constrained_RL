from dataclasses import dataclass
from typing import Optional, Union, List

@dataclass
class TrainingParams:
    
  seed:               int
  samples_per_epoch:  int
  epochs:             int
  patience:           int
  tau:                float
  epsilon:            float
  smooth_lambda:      float

@dataclass
class EntropyParams:

  alpha:              Union[str, float]
  init_alpha:         Optional[float]
  target_alpha:       Union[str, float]
  alpha_betas:        List[float]
  alpha_lr:           float

@dataclass
class CostParams:

  fixed_cost_penalty: Optional[float]
  init_beta:          Optional[float]
  cost_limit:         Optional[float]
  target_cost:        Union[str, float]
  beta_betas:         List[float]
  beta_lr:            float
  cost_lr_scale:      float
  risk_level:         float
  damp_scale:         float

@dataclass
class EnvironmentParams:

  env:                Union[str, List[str]]
  lidar_num_bins:     int
  lidar_max_dist:     Optional[float]
  lidar_exp_gain:     float
  lidar_type:         str

@dataclass
class UtilitiesParams:

  early_stopping:     bool
  use_profiler:       bool
  record_video:       bool
  record_epochs:      int
  fast_dev_run:       bool

@dataclass
class Params:
    
  agent:              classmethod
  training_params:    TrainingParams
  entropy_params:     EntropyParams
  cost_params:        CostParams
  environment_params: EnvironmentParams
  utilities_params:   UtilitiesParams

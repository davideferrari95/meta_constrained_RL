from dataclasses import dataclass
from typing import Optional, Union, List

@dataclass
class TrainingParams:
    
  env:                Union[str, List[str]]
  samples_per_epoch:  int
  epochs:             int
  patience:           int
  tau:                float

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
class UtilitiesParams:

  early_stopping:     bool
  record_video:       bool
  fast_dev_run:       bool

@dataclass
class Params:
    
    agent:              classmethod
    training_params:    TrainingParams
    entropy_params:     EntropyParams
    cost_params:        CostParams
    utilities_params:   UtilitiesParams

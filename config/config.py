from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class TrainingParams:
    
  env:                str
  samples_per_epoch:  int
  epochs:             int
  patience:           int
  tau:                float

@dataclass
class EntropyParams:

  alpha:              Union[str, float]
  target_alpha:       Union[str, float]
  init_alpha:         Optional[float]

@dataclass
class CostParams:

  fixed_cost_penalty: Optional[float]
  cost_constraint:    Optional[float]
  cost_limit:         Optional[float]

@dataclass
class UtilitiesParams:

  early_stopping:     bool
  record_video:       bool
  fast_dev_run:       bool

@dataclass
class Params:
    
    training_params:    TrainingParams
    entropy_params:     EntropyParams
    cost_params:        CostParams
    utilities_params:   UtilitiesParams

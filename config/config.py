from dataclasses import dataclass
from typing import Optional, Union, List

@dataclass
class TrainingParams:
    
  seed:               int
  samples_per_epoch:  int
  min_epochs:         int
  max_epochs:         int
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
class SafetyParams:

  unsafe_experience:  bool    # Activate or Not UnSafe Experience Saving in ReplayBuffer

@dataclass
class EnvironmentParams:

  # Task Configuration
  env_name:           Union[str, List[str]]
  robot_base:         str
  task:               str

  # Rewards
  reward_distance:    float   # Dense reward multiplied by the distance moved to the goal
  reward_goal:        float   # Sparse reward for being inside the goal area
  penalty_step:		    float   # Reward Every Non-Goal Step

  # World Limits
  world_limits:       List[float]   # Soft world limits (min X, min Y, max X, max Y)
  placements_extents: List[float]   # Placement limits (min X, min Y, max X, max Y)

  # Activation Bool
  observe_goal_lidar: bool    # Enable Goal Lidar
  observe_buttons:    bool    # Lidar observation of button object positions
  observe_hazards:    bool    # Enable Hazard Lidar
  observe_vases:      bool    # Observe the vector from agent to vases
  observe_pillars:    bool    # Lidar observation of pillar object positions
  observe_gremlins:   bool    # Gremlins are observed with lidar-like space
  observe_walls:      bool    # Observe the walls with a lidar space
  observe_world_lim:  bool    # Observe world limits

  constrain_hazards:  bool    # Penalty Entering in Hazards
  constrain_vases:    bool    # Constrain robot from touching objects
  constrain_pillars:  bool    # Immovable obstacles in the environment
  constrain_gremlins: bool    # Moving objects that must be avoided
  constrain_buttons:  bool    # Penalize pressing incorrect buttons

  # Lidar Config
  lidar_num_bins:     int               # Number of Lidar Dots
  lidar_max_dist:     Optional[float]   # Maximum distance for lidar sensitivity (if None, exponential distance)
  lidar_exp_gain:     float             # Scaling factor for distance in exponential distance lidar
  lidar_type:         str               # 'pseudo', 'natural', see self.obs_lidar()

  # Goal Config
  goal_size:          float   # Size of Goal (0.3)
  goal_keepout:       float   # Min Spawn Distance to Hazard

  # Hazard Config
  hazards_num:        int     # Number of Hazards
  hazards_size:       float   # Size of Hazard (0.2)
  hazards_keepout:    float   # Min Spawn Distance to Hazard
  hazards_cost:       float   # Cost (per step) for violating the constraint

  # Vases Config
  vases_num:           int    # Number of vases in the world
  vases_contact_cost:  float  # Cost (per step) for being in contact with a vase
  vases_displace_cost: float  # Cost (per step) per meter of displacement for a vase
  vases_velocity_cost: float  # Cost (per step) per m/s of velocity for a vase

  # Robot Starting Location
  robot_locations:    Optional[List[float]]   # Explicitly Place Robot XY Coordinates
  robot_rot:          Optional[float]         # Override Robot Starting Angle

  # HardCoded Location of Goal and Hazards
  goal_locations:     Optional[List[float]]         # Explicitly Place Goal XY Coordinates
  hazards_locations:  Optional[List[List[float]]]   # Explicitly Place Hazards XY Coordinates

  # Robot Sensors
  sensors_obs:        List[str]   # Mujoco Sensors

  # Threshold and Penalty
  stuck_threshold:    float
  stuck_penalty:      float
  safety_threshold:   float

  # Test and Violation Environments
  test_environment:       bool
  violation_environment:  bool
  test_env_epochs:        int
  violation_env_epochs:   int
  test_episode_number:    int

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
  safe_params:        SafetyParams
  environment_params: EnvironmentParams
  utilities_params:   UtilitiesParams

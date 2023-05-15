from dataclasses import dataclass
from typing import Optional, Union, List

@dataclass
class TrainingParams:
    
  initial_samples:    int
  min_epochs:         int
  max_epochs:         int
  precision:          int
  early_stopping:     bool
  early_stop_metric:  str
  patience:           int
  record_video:       bool
  record_epochs:      int
  profiler:           str
  compilation_mode:   str
  torch_compilation:  bool
  fast_dev_run:       bool

@dataclass
class EnvironmentParams:

  # Task Configuration
  env_name:           Union[str, List[str]]
  seed:               int

  robot_base:         str
  task:               str

  # Rewards
  reward_distance:    float   # Dense reward multiplied by the distance moved to the goal
  reward_goal:        float   # Sparse reward for being inside the goal area

  # World Spawn Limits
  placements_extents: List[float]   # Placement limits (min X, min Y, max X, max Y)

  # Activation Bool
  observe_goal_lidar: bool    # Enable Goal Lidar
  observe_buttons:    bool    # Lidar observation of button object positions
  observe_hazards:    bool    # Enable Hazard Lidar
  observe_vases:      bool    # Observe the vector from agent to vases
  observe_pillars:    bool    # Lidar observation of pillar object positions
  observe_gremlins:   bool    # Gremlins are observed with lidar-like space
  observe_walls:      bool    # Observe the walls with a lidar space

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
  test_unconstrained:     bool
  test_constrained:       bool
  violation_environment:  bool
  test_env_epochs:        int
  violation_env_epochs:   int
  test_episode_number:    int

@dataclass
class Params:

  agent:              classmethod
  training_params:    TrainingParams
  environment_params: EnvironmentParams

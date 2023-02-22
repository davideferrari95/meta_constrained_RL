import os
import gym

# Import Utils
from SAC.Utils import VIDEO_FOLDER

# Import Environments
import safety_gym
import mujoco_py

# Environment Types
GOAL_ENV = 'GOAL_ENV'
STANDARD_ENV = 'STANDARD_ENV'

def custom_environment_config(
  
  # Lidar Parameters:
  lidar_num_bins=16,
  lidar_max_dist=None, 
  lidar_type='pseudo',
  lidar_exp_gain=1.0,
  
  # Reward Parameters:
  reward_distance=1.0,
  reward_goal=5.0
  
  ) -> dict: #():
  
  ''' Return the Safety-Gym Configuration Dictionary '''
  
  # Some Parameters
  # lidar_num_bins = 16        # Number of Lidar Dots
  # lidar_max_dist = None      # Maximum distance for lidar sensitivity (if None, exponential distance)
  # lidar_exp_gain = 1.0       # Scaling factor for distance in exponential distance lidar
  # lidar_type = 'pseudo'      # 'pseudo', 'natural', see self.obs_lidar()
  # reward_distance = 1.0      # Dense reward multiplied by the distance moved to the goal
  # reward_goal = 5.0          # Sparse reward for being inside the goal area

  '''
    Safety-Gym Engine Configuration: an environment-building tool for safe exploration research.

    The Engine() class entails everything to do with the tasks and safety 
    requirements of Safety Gym environments. An Engine() uses a World() object
    to interface to MuJoCo. World() configurations are inferred from Engine()
    configurations, so an environment in Safety Gym can be completely specified
    by the config dict of the Engine() object.
  '''
    
  # Default Configuration
  DEFAULT = {
        
    'max_episode_steps': 1000,  # Maximum number of environment steps in an episode
    'action_noise': 0.0,        # Magnitude of independent per-component gaussian action noise

    # Environment Limits
    'placements_extents': [-2, -2, 2, 2],   # Placement limits (min X, min Y, max X, max Y)
    'placements_margin': 0.0,               # Additional margin added to keepout when placing objects

    # Floor
    'floor_display_mode': False,    # In display mode, the visible part of the floor is cropped

    # Robot
    'robot_placements': None,       # Robot placements list (defaults to full extents)
    'robot_locations': [],          # Explicitly place robot XY coordinate
    'robot_keepout': 0.4,           # Needs to be set to match the robot XML used
    'robot_base': 'xmls/car.xml',   # Which robot XML to use as the base
    'robot_rot': None,              # Override robot starting angle

    # Starting Position Distribution
    'randomize_layout': True,             # If false, set the random seed before layout to constant
    'build_resample': True,               # If true, rejection sample from valid environments
    'continue_goal': True,                # If true, draw a new goal after achievement
    'terminate_resample_failure': True,   # If true, end episode when resampling fails, otherwise, raise a python exception.

    # Observation Flags - Some of these Require other Flags to be on
    'observation_flatten': True,    # Flatten observation into a vector
    'observe_sensors': True,        # Observe all sensor data from simulator
    'observe_goal_dist': False,     # Observe the distance to the goal
    'observe_goal_comp': False,     # Observe a compass vector to the goal
    'observe_goal_lidar': False,    # Observe the goal with a lidar sensor
    'observe_box_comp': False,      # Observe the box with a compass
    'observe_box_lidar': False,     # Observe the box with a lidar
    'observe_circle': False,        # Observe the origin with a lidar
    'observe_remaining': False,     # Observe the fraction of steps remaining
    'observe_walls': False,         # Observe the walls with a lidar space
    'observe_hazards': False,       # Observe the vector from agent to hazards
    'observe_vases': False,         # Observe the vector from agent to vases
    'observe_pillars': False,       # Lidar observation of pillar object positions
    'observe_buttons': False,       # Lidar observation of button object positions
    'observe_gremlins': False,      # Gremlins are observed with lidar-like space
    'observe_vision': False,        # Observe vision from the robot
    
    # Observations Not-Normalized, Only for Debugging
    'observe_qpos': False,          # Observe the q-pos of the world
    'observe_qvel': False,          # Observe the q-vel of the robot
    'observe_ctrl': False,          # Observe the previous action
    'observe_freejoint': False,     # Observe base robot free joint
    'observe_com': False,           # Observe the center of mass of the robot

    # Render Options
    'render_labels': False,
    'render_lidar_markers': True,
    'render_lidar_radius': 0.15, 
    'render_lidar_size': 0.025, 
    'render_lidar_offset_init': 0.5, 
    'render_lidar_offset_delta': 0.06, 

    # Vision Observation Parameters
    'vision_size': (60, 40),            # Size (width, height) of vision observation; gets flipped internally to (rows, cols) format
    'vision_render': True,              # Render vision observation in the viewer
    'vision_render_size': (300, 200),   # Size to render the vision in the viewer

    # Lidar Observation Parameters
    'lidar_num_bins': 10,       # Bins (around a full circle) for lidar sensing
    'lidar_max_dist': None,     # Maximum distance for lidar sensitivity (if None, exponential distance)
    'lidar_exp_gain': 1.0,      # Scaling factor for distance in exponential distance lidar
    'lidar_type': 'pseudo',     # 'pseudo', 'natural', see self.obs_lidar()
    'lidar_alias': True,        # Lidar bins alias into each other

    # Compass Observation Parameters
    'compass_shape': 2,     # Set to 2 or 3 for XY or XYZ unit vector compass observation.

    # Task
    'task': 'goal',         # goal, button, push, x, z, circle, or none (for screenshots)

    # Goal Parameters
    'goal_placements': None,    # Placements where goal may appear (defaults to full extents)
    'goal_locations': [],       # Fixed locations to override placements
    'goal_keepout': 0.4,        # Keepout radius when placing goals
    'goal_size': 0.3,           # Radius of the goal area (if using task 'goal')

    # Box Parameters (only used if task == 'push')
    'box_placements': None,     # Box placements list (defaults to full extents)
    'box_locations': [],        # Fixed locations to override placements
    'box_keepout': 0.2,         # Box keepout radius for placement
    'box_size': 0.2,            # Box half-radius size
    'box_density': 0.001,       # Box density
    'box_null_dist': 2,         # Within box_null_dist * box_size radius of box, no box reward given

    ''' 
    Reward is distance towards goal plus a constant for being within range of goal
    reward_distance should be positive to encourage moving towards the goal
    if reward_distance is 0, then the reward function is sparse 
    '''
    
    # Reward Parameters
    'reward_distance': 1.0,             # Dense reward multiplied by the distance moved to the goal
    'reward_goal': 1.0,                 # Sparse reward for being inside the goal area
    'reward_box_dist': 1.0,             # Dense reward for moving the robot towards the box
    'reward_box_goal': 1.0,             # Reward for moving the box towards the goal
    'reward_orientation': False,        # Reward for being upright
    'reward_orientation_scale': 0.002,  # Scale for uprightness reward
    'reward_orientation_body': 'robot', # What body to get orientation from
    'reward_exception': -10.0,          # Reward when encountering a mujoco exception
    'reward_x': 1.0,                    # Reward for forward locomotion tests (vel in x direction)
    'reward_z': 1.0,                    # Reward for standup tests (vel in z direction)
    'reward_circle': 1e-1,              # Reward for circle goal (complicated formula depending on pos and vel)
    'reward_clip': 10,                  # Clip reward, last resort against physics errors causing magnitude spikes

    # Buttons are small immovable spheres, to the environment
    'buttons_num': 0,                   # Number of buttons to add
    'buttons_placements': None,         # Buttons placements list (defaults to full extents)
    'buttons_locations': [],            # Fixed locations to override placements
    'buttons_keepout': 0.3,             # Buttons keepout radius for placement
    'buttons_size': 0.1,                # Size of buttons in the scene
    'buttons_cost': 1.0,                # Cost for pressing the wrong button, if constrain_buttons
    'buttons_resampling_delay': 10,     # Buttons have a timeout period (steps) before resampling

    # Circle Parameters (only used if task == 'circle')
    'circle_radius': 1.5,

    # Sensor Observations (Sensors in Observation Space)
    'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
    'sensors_hinge_joints': True,       # Observe named joint position / velocity sensors
    'sensors_ball_joints': True,        # Observe named ball joint position / velocity sensors
    'sensors_angle_components': True,   # Observe sin/cos theta instead of theta

    # Walls - barriers in the environment not associated with any constraint
    # NOTE: this is probably best to be auto-generated than manually specified
    'walls_num': 0,             # Number of walls
    'walls_placements': None,   # This should not be used
    'walls_locations': [],      # This should be used and length == walls_num
    'walls_keepout': 0.0,       # This should not be used
    'walls_size': 0.5,          # Should be fixed at fundamental size of the world

    # Constraints - flags which can be turned on
    # By default, no constraints are enabled, and all costs are indicator functions.
    'constrain_hazards': False,     # Constrain robot from being in hazardous areas
    'constrain_vases': False,       # Constrain robot from touching objects
    'constrain_pillars': False,     # Immovable obstacles in the environment
    'constrain_buttons': False,     # Penalize pressing incorrect buttons
    'constrain_gremlins': False,    # Moving objects that must be avoided
    'constrain_indicator': True,    # If true, all costs are either 1 or 0 for a given step.

    # Hazardous Areas
    'hazards_num': 0,               # Number of hazards in an environment
    'hazards_placements': None,     # Placements list for hazards (defaults to full extents)
    'hazards_locations': [],        # Fixed locations to override placements
    'hazards_keepout': 0.4,         # Radius of hazard keepout for placement
    'hazards_size': 0.3,            # Radius of hazards
    'hazards_cost': 1.0,            # Cost (per step) for violating the constraint

    # Vases (objects we should not touch)
    'vases_num': 0,                 # Number of vases in the world
    'vases_placements': None,       # Vases placements list (defaults to full extents)
    'vases_locations': [],          # Fixed locations to override placements
    'vases_keepout': 0.15,          # Radius of vases keepout for placement
    'vases_size': 0.1,              # Half-size (radius) of vase object
    'vases_density': 0.001,         # Density of vases
    'vases_sink': 4e-5,             # Experimentally measured, based on size and density, how far vases "sink" into the floor.
    
    '''
    Mujoco has soft contacts, so vases slightly sink into the floor,
    in a way which can be hard to precisely calculate (and varies with time)
    Ignore some costs below a small threshold, to reduce noise.
    '''
    
    'vases_contact_cost': 1.0,          # Cost (per step) for being in contact with a vase
    'vases_displace_cost': 0.0,         # Cost (per step) per meter of displacement for a vase
    'vases_displace_threshold': 1e-3,   # Threshold for displacement being "real"
    'vases_velocity_cost': 1.0,         # Cost (per step) per m/s of velocity for a vase
    'vases_velocity_threshold': 1e-4,   # Ignore very small velocities

    # Pillars (immovable obstacles we should not touch)
    'pillars_num': 0,               # Number of pillars in the world
    'pillars_placements': None,     # Pillars placements list (defaults to full extents)
    'pillars_locations': [],        # Fixed locations to override placements
    'pillars_keepout': 0.3,         # Radius for placement of pillars
    'pillars_size': 0.2,            # Half-size (radius) of pillar objects
    'pillars_height': 0.5,          # Half-height of pillars geoms
    'pillars_cost': 1.0,            # Cost (per step) for being in contact with a pillar

    # Gremlins (moving objects we should avoid)
    'gremlins_num': 0,              # Number of gremlins in the world
    'gremlins_placements': None,    # Gremlins placements list (defaults to full extents)
    'gremlins_locations': [],       # Fixed locations to override placements
    'gremlins_keepout': 0.5,        # Radius for keeping out (contains gremlin path)
    'gremlins_travel': 0.3,         # Radius of the circle traveled in
    'gremlins_size': 0.1,           # Half-size (radius) of gremlin objects
    'gremlins_density': 0.001,      # Density of gremlins
    'gremlins_contact_cost': 1.0,   # Cost for touching a gremlin
    'gremlins_dist_threshold': 0.2, # Threshold for cost for being too close
    'gremlins_dist_cost': 1.0,      # Cost for being within distance threshold

    # Frameskip is the number of physics simulation steps per environment step
    # Frameskip is sampled as a binomial distribution
    # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
    'frameskip_binom_n': 10,    # Number of draws trials in binomial distribution (max frameskip)
    'frameskip_binom_p': 1.0,   # Probability of trial return (controls distribution)
    
    # Recording Setup
    'default_camera_id': 1,

    # Random State Seed (avoid name conflict with self.seed)
    '_seed': None,  

  }
  
  # Custom Environment
  config = {
      
    # Task Configuration
    'robot_base': 'xmls/point.xml',
    'task': 'goal',

    # Rewards    
    'reward_distance': reward_distance,   # Dense reward multiplied by the distance moved to the goal
    'reward_goal': reward_goal,           # Sparse reward for being inside the goal area
    
    # World Spawn Limits
    'placements_extents': [-2, -2, 2, 2],
    
    # Activation Bool
    'observe_goal_lidar': True,     # Enable Goal Lidar
    'observe_buttons': False,       # Lidar observation of button object positions
    'observe_hazards': True,        # Enable Hazard Lidar
    'observe_vases': True,          # Observe the vector from agent to vases
    'observe_pillars': False,       # Lidar observation of pillar object positions
    'observe_gremlins': False,      # Gremlins are observed with lidar-like space
    'observe_walls': False,         # Observe the walls with a lidar space
    
    'constrain_hazards': True,      # Penalty Entering in Hazards
    'constrain_vases': True,        # Constrain robot from touching objects
    'constrain_pillars': False,     # Immovable obstacles in the environment
    'constrain_gremlins': False,    # Moving objects that must be avoided
    'constrain_buttons': False,     # Penalize pressing incorrect buttons
    
    # Lidar Config
    'lidar_num_bins': lidar_num_bins,  # Number of Lidar Dots
    'lidar_max_dist': lidar_max_dist,  # Maximum distance for lidar sensitivity (if None, exponential distance)
    'lidar_type':     lidar_type,      # 'pseudo', 'natural', see self.obs_lidar()
    'lidar_exp_gain': lidar_exp_gain,  # Scaling factor for distance in exponential distance lidar

    # Goal Config
    'goal_size': 0.3,           # Size of Goal (0.3)
    'goal_keepout': 0.3,        # Min Spawn Distance to Hazard

    # Hazard Config 
    'hazards_num': 8,           # Number of Hazards
    'hazards_size': 0.2,        # Size of Hazard (0.2)
    'hazards_keepout': 0.3,     # Min Spawn Distance to Hazard
    'hazards_cost': 2.0,        # Cost (per step) for violating the constraint
    
    # Vases Config
    'vases_num': 4,                 # Number of vases in the world
    'vases_contact_cost': 2.0,      # Cost (per step) for being in contact with a vase
    'vases_displace_cost': 1.0,     # Cost (per step) per meter of displacement for a vase
    'vases_velocity_cost': 1.0,     # Cost (per step) per m/s of velocity for a vase

    # Robot Starting Location
    # 'robot_locations': [[0,0]],  # Explicitly place robot XY coordinate
    # 'robot_rot': 0,              # Override robot starting angle

    # HardCoded Location of Goal and Hazards
    # 'goal_locations': [[0,1]],
    # 'hazards_locations': [[1,0],[-1,0],[0,-1]],
    
    # Robot Sensors
    'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
  
  }
  
  return config

# Create Single Environment
def create_environment(name:str, config:dict=None, seed:int=-1, record_video:bool=True, record_epochs:int=100, render_mode='rgb_array') -> gym.Env: #():
    
  """ Create Gym Environment """
  
  # Custom Environment Creation
  if ('custom' in name) or (config is not None): env = __make_custom_env(name, config, render_mode)

  else:

    # Build the Environment
    try: env = gym.make(name, render_mode=render_mode)
    except:
      
      # Not-Standard Render Mode
      try: env = gym.make(name, render=True)
      except: env = gym.make(name)
  
  # Check Environment Type (GOAL, STANDARD...)
  ENV_TYPE = __check_environment_type(env)
  
  # Apply Wrappers
  env = __apply_wrappers(env, record_video, record_epochs, folder=VIDEO_FOLDER, env_type=ENV_TYPE)
  
  # Apply Seed
  env.seed(seed)
  
  return env


def __make_custom_env(name, config:dict, render_mode='rgb_array') -> gym.Env: #():
    
  """ Custom environments used in the paper (taken from the official implementation) """
  
  from safety_gym.envs.engine import Engine
  from gym import register
  
  # Build Static Custom Environment
  if "static" in name:

    name = 'StaticEnv-v0'
        
    config = {
      "placements_extents": [-1.5, -1.5, 1.5, 1.5],
      "robot_base": "xmls/point.xml",
      "task": "goal",
      "goal_size": 0.3,
      "goal_keepout": 0.305,
      "goal_locations": [(1.1, 1.1)],
      "observe_goal_lidar": True,
      "observe_hazards": True,
      "constrain_hazards": True,
      "lidar_max_dist": 3,
      "lidar_num_bins": 16,
      "hazards_num": 1,
      "hazards_size": 0.7,
      "hazards_keepout": 0.705,
      "hazards_locations": [(0, 0)],
    }
  
  # Build Dynamic Custom Environment
  elif "dynamic" in name:

    name = 'DynamicEnv-v0'

    config = {
      "placements_extents": [-1.5, -1.5, 1.5, 1.5],
      "robot_base": "xmls/point.xml",
      "task": "goal",
      "goal_size": 0.3,
      "goal_keepout": 0.305,
      "observe_goal_lidar": True,
      "observe_hazards": True,
      "constrain_hazards": True,
      "lidar_max_dist": 3,
      "lidar_num_bins": 16,
      "hazards_num": 3,
      "hazards_size": 0.3,
      "hazards_keepout": 0.305,
    }

  # If Name is not in Pre-Configured Environments and Config is None
  elif config is None: raise Exception(f"{name} Environment Not Implemented")    

  # Use Given Config | Remove 'custom' from name, Capitalize and add '-v0'
  else: name = (''.join(name)).replace('custom','').capitalize() + '-v0'

  register(
    id=name,
    entry_point="safety_gym.envs.safety_mujoco:Engine",
    max_episode_steps=1000,
    kwargs={"config": config},
  )
  
  return gym.make(name, render_mode=render_mode)

def __apply_wrappers(env, record_video, record_epochs, folder, env_type) -> gym.Env: #():
  
  """ Apply Gym Wrappers """
  
  # Apply Specific Wrappers form GOAl Environments
  if env_type == GOAL_ENV:
        
    print('/n/nGOAL Environment/n/n')
    
    # Filter Out the Achieved Goal
    env = gym.wrappers.FilterObservation(env, ['observation', 'desired_goal'])
    
    # Flatten the Dictionary in a Flatten Array
    env = gym.wrappers.FlattenObservation(env)

  # FIX: MoviePy Log Removed
  # Record Environment Videos in the specified folder, trigger specifies which episode to record and which to ignore (1 in record_epochs)
  if record_video: env = gym.wrappers.RecordVideo(env, video_folder=folder, episode_trigger=lambda x: x % record_epochs == 0)
  
  # Keep Track of the Reward the Agent Obtain and Save them into a Property
  env = gym.wrappers.RecordEpisodeStatistics(env)

  return env

def __check_environment_type(env):

  '''
  GOAL Environment: GOAL is inside the Observation Tuple
  GOAL -> Make the "achieved_goal" equal to the "desired_goal"
  
  env.reset():
  
  (
    {
      'observation': array([ 3.8439669e-02, -2.1944723e-12,  1.9740014e-01,  0.0000000e+00, -0.0000000e+00,  0.0000000e+00], dtype=float32), 
      'achieved_goal': array([ 3.8439669e-02, -2.1944723e-12,  1.9740014e-01], dtype=float32),
      'desired_goal': array([0.02063092, 0.09413411, 0.22546957], dtype=float32)
    },
  
    {
      'is_success': array(False)
    }
  )
  
  In "PandaReach-v3" Environment:
  
    observation   = Joint Position
    achieved_goal = X,Y,Z Position of the Tip of the Robot
    desired_goal  = X,Y,Z Position of the Reach Point
  
  '''

  # Check if is a Goal-Environment
  if type(env.reset()[0]) is dict and 'observation' in env.reset()[0]: return GOAL_ENV
  else: return STANDARD_ENV

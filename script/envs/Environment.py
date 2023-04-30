import sys, os
import numpy as np
from typing import Optional

# Import Utils
from utils.Utils import FOLDER, VIDEO_FOLDER, VIOLATIONS_FOLDER, TEST_FOLDER
from utils.Utils import video_rename

# Import Parameters Class
sys.path.append(FOLDER)
from config.config import EnvironmentParams

# Import Environments
import gym, safety_gym
import mujoco_py

# Environment Types
GOAL_ENV = 'GOAL_ENV'
STANDARD_ENV = 'STANDARD_ENV'

def custom_environment_config(config:EnvironmentParams) -> dict: #():

  ''' Return the Safety-Gym Configuration Dictionary '''

  '''
    Safety-Gym Engine Configuration: an environment-building tool for safe exploration research.

    The Engine() class entails everything to do with the tasks and safety 
    requirements of Safety Gym environments. An Engine() uses a World() object
    to interface to MuJoCo. World() configurations are inferred from Engine()
    configurations, so an environment in Safety Gym can be completely specified
    by the config dict of the Engine() object.
  '''

  # Return None if not Custom Environment
  if not 'custom' in config.env_name: return config.env_name, None

  # Remove Penalty when Get the Goal Reward
  config.reward_goal += config.penalty_step

  # Custom Environment
  env_config = {

    # Task Configuration
    'robot_base': config.robot_base,
    'task': config.task,

    # Rewards
    'reward_distance': config.reward_distance,   # Dense reward multiplied by the distance moved to the goal
    'reward_goal':     config.reward_goal,       # Sparse reward for being inside the goal area

    # World Spawn Limits
    'world_limits':       config.world_limits,          # Soft world limits (min X, min Y, max X, max Y)
    'placements_extents': config.placements_extents,    # Placement limits (min X, min Y, max X, max Y)

    # Activation Bool
    'observe_goal_lidar': config.observe_goal_lidar,    # Enable Goal Lidar
    'observe_buttons':    config.observe_buttons,       # Lidar observation of button object positions
    'observe_hazards':    config.observe_hazards,       # Enable Hazard Lidar
    'observe_vases':      config.observe_vases,         # Observe the vector from agent to vases
    'observe_pillars':    config.observe_pillars,       # Lidar observation of pillar object positions
    'observe_gremlins':   config.observe_gremlins,      # Gremlins are observed with lidar-like space
    'observe_walls':      config.observe_walls,         # Observe the walls with a lidar space
    'observe_world_lim':  config.observe_world_lim,     # Observe world limits

    'constrain_hazards':  config.constrain_hazards,     # Penalty Entering in Hazards
    'constrain_vases':    config.constrain_vases,       # Constrain robot from touching objects
    'constrain_pillars':  config.constrain_pillars,     # Immovable obstacles in the environment
    'constrain_gremlins': config.constrain_gremlins,    # Moving objects that must be avoided
    'constrain_buttons':  config.constrain_buttons,     # Penalize pressing incorrect buttons

    # Lidar Config
    'lidar_num_bins': config.lidar_num_bins,    # Number of Lidar Dots
    'lidar_max_dist': config.lidar_max_dist,    # Maximum distance for lidar sensitivity (if None, exponential distance)
    'lidar_exp_gain': config.lidar_exp_gain,    # Scaling factor for distance in exponential distance lidar
    'lidar_type':     config.lidar_type,        # 'pseudo', 'natural', see self.obs_lidar()

    # Goal Config
    'goal_size':    config.goal_size,           # Size of Goal (0.3)
    'goal_keepout': config.goal_keepout,        # Min Spawn Distance to Hazard

    # Hazard Config
    'hazards_num':     config.hazards_num,      # Number of Hazards
    'hazards_size':    config.hazards_size,     # Size of Hazard (0.2)
    'hazards_keepout': config.hazards_keepout,  # Min Spawn Distance to Hazard
    'hazards_cost':    config.hazards_cost,     # Cost (per step) for violating the constraint

    # Vases Config
    'vases_num':           config.vases_num,              # Number of vases in the world
    'vases_contact_cost':  config.vases_contact_cost,     # Cost (per step) for being in contact with a vase
    'vases_displace_cost': config.vases_displace_cost,    # Cost (per step) per meter of displacement for a vase
    'vases_velocity_cost': config.vases_velocity_cost,    # Cost (per step) per m/s of velocity for a vase

    # Robot Starting Location
    # 'robot_locations': config.robot_locations,    # Explicitly Place Robot XY Coordinates
    # 'robot_rot':       config.robot_rot,          # Override Robot Starting Angle

    # HardCoded Location of Goal and Hazards
    # 'goal_locations':    config.goal_locations,     # Explicitly Place Goal XY Coordinates
    # 'hazards_locations': config.hazards_locations,  # Explicitly Place Hazards XY Coordinates

    # Robot Sensors (Mujoco Sensors)
    'sensors_obs': config.sensors_obs,

  }

  return config.env_name, env_config


# Create Vectorized Environments
def create_vectorized_environment(name:str, config:dict=None, env_num:int=10, seed:int=-1, 
                                  record_video:bool=True, record_epochs:int=100, 
                                  render_mode='rgb_array', apply_wrappers:bool=True,
                                  environment_type:Optional[str]=None, env_epochs:int=1) -> gym.vector.SyncVectorEnv: #():

  " Create Vectorized Gym Environment "

  # Create Vectorized Environment (Record Video Only for First Environment)
  envs = gym.vector.SyncVectorEnv([create_environment(name, config, seed + i, record_video if i == 0 else False, 
                                    record_epochs, render_mode, apply_wrappers, environment_type, env_epochs) for i in range(env_num)])

  return envs

# Create Single Environment
def create_environment(name:str, config:dict=None, seed:int=-1, 
                       record_video:bool=True, record_epochs:int=100, 
                       render_mode='rgb_array', apply_wrappers:bool=True,
                       environment_type:Optional[str]=None, env_epochs:int=1) -> gym.Env: #():

  """ Create Gym Environment """

  def make_env():

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
    if   environment_type == 'test'      and record_video: env = gym.wrappers.RecordVideo(env, video_folder=TEST_FOLDER,       episode_trigger=lambda x: x % env_epochs == 0, name_prefix='test_')
    elif environment_type == 'violation' and record_video: env = gym.wrappers.RecordVideo(env, video_folder=VIOLATIONS_FOLDER, episode_trigger=lambda x: x % env_epochs == 0, name_prefix='new_')
    elif apply_wrappers: env = __apply_wrappers(env, record_video, record_epochs, folder=VIDEO_FOLDER, env_type=ENV_TYPE)

    # TODO: Apply Seed -> In env.reset() ?
    # env.seed(seed)

    return env

  return make_env

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
  else: name = (''.join(name)).replace('custom','') + '-v0'

  # Check if Environment is Already Registered
  if name not in gym.envs.registry:

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
  if record_video: env = gym.wrappers.RecordVideo(env, video_folder=folder, episode_trigger=lambda x: x % record_epochs == 0 and x != 0)

  # Keep Track of the Reward the Agent Obtain and Save them into a Property
  env = gym.wrappers.RecordEpisodeStatistics(env)

  # Preprocess the Environment
  env = gym.wrappers.ClipAction(env)
  env = gym.wrappers.NormalizeObservation(env)
  env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10,10))
  env = gym.wrappers.NormalizeReward(env)
  env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10,10))

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

def record_violation_episode(env:gym.Env, seed:int, action_list, current_epoch:int):

  # Return if Violation Folder Not Exist
  if not os.path.exists(VIOLATIONS_FOLDER): return

  # Reset Environment with Seeding
  env.reset(seed=seed)

  # Apply All Action in `action_list` Buffer
  for action in action_list: env.step(action)

  # Rename Video with Current Epoch Name
  for filename in os.listdir(VIOLATIONS_FOLDER):
    if filename.startswith('new_'): video_rename(VIOLATIONS_FOLDER, filename, f'violation-episode-{current_epoch}')

def rename_test_episodes(prefix=''):

  # Rename Video with Current Epoch Name
  for filename in os.listdir(TEST_FOLDER):

    # Check if .mp4 or .json
    if filename.endswith('.mp4'):    number = filename[14:-4]
    elif filename.endswith('.json'): number = filename[14:-10]

    if filename.startswith('test_'): video_rename(TEST_FOLDER, filename, f'{prefix}-test-episode-{number}')

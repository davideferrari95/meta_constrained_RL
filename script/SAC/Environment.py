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

# Create Single Environment
def create_environment(name, seed=-1, record_video=True, record_epochs=100, render_mode='rgb_array') -> gym.Env: #():
    
  """ Create Gym Environment """
  
  # Custom Environment Creation
  if 'custom' in name: env = __make_custom_env(name, render_mode)

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


def __make_custom_env(name, render_mode='rgb_array') -> gym.Env: #():
    
  """ Custom environments used in the paper (taken from the official implementation) """
  
  from safety_gym.envs.engine import Engine
  from gym import register
  
  # Build Static Custom Environment
  if "static" in name:
        
    config1 = {
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
    
    register(
      id="StaticEnv-v0",
      entry_point="safety_gym.envs.safety_mujoco:Engine",
      max_episode_steps=1000,
      kwargs={"config": config1},
    )
      
    env = gym.make("StaticEnv-v0", render_mode=render_mode)
  
  # Build Dynamic Custom Environment
  elif "dynamic" in name:
      
    config2 = {
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
    
    register(
      id="DynamicEnv-v0",
      entry_point="safety_gym.envs.safety_mujoco:Engine",
      max_episode_steps=1000,
      kwargs={"config": config2},
    )
    
    env = gym.make("DynamicEnv-v0", render_mode=render_mode)
    
  else: raise Exception(f"{name} Environment Not Implemented")    

  return env

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

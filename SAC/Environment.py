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
def create_environment(name, record_video=True, render_mode='rgb_array') -> gym.Env:
  return __create_environment(name, record_video, render_mode)

def __create_environment(name, record_video, render_mode='rgb_array'):
    
    # Build the Environment
    try: env = gym.make(name, render_mode=render_mode)
    except:
      
      # Not-Standard Render Mode 
      try: env = gym.make(name, render=True)
      except: env = gym.make(name)
    
    # Check Environment Type (GOAL, STANDARD...)
    ENV_TYPE = __check_environment_type(env)
    
    # Apply Wrappers
    env = __apply_wrappers(env, record_video, folder=VIDEO_FOLDER, env_type=ENV_TYPE)
    
    return env

def __apply_wrappers(env, record_video, folder, env_type):
      
  # Apply Specific Wrappers form GOAl Environments
  if env_type == GOAL_ENV:
        
        print('/n/nGOAL Environment/n/n')
        
        # Filter Out the Achieved Goal
        env = gym.wrappers.FilterObservation(env, ['observation', 'desired_goal'])
        
        # Flatten the Dictionary in a Flatten Array
        env = gym.wrappers.FlattenObservation(env)

  # FIX: MoviePy Log Removed
  # Record Environment Videos in the specified folder, trigger specifies which episode to record and which to ignore (1 in 50)
  if record_video: env = gym.wrappers.RecordVideo(env, video_folder=folder, episode_trigger=lambda x: x % 100 == 0)
  
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

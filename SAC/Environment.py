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

# Create Multiple Parallel Environments
def create_parallel_environment(name, num_envs=100, render_mode='rgb_array') -> gym.vector.VectorEnv:
  return __create_parallel_environment(name, num_envs, render_mode)

def __create_parallel_environment(name, num_envs=100, render_mode='rgb_array'):
    
    # Build the Environment
    try: envs = gym.vector.make(name, num_envs=num_envs, render_mode=render_mode)
    except:
      
      # Not-Standard Render Mode 
      try: envs = gym.vector.make(name, num_envs=num_envs, render=True)
      except: envs = gym.vector.make(name, num_envs=num_envs)
    
    # Check Environment Type (GOAL, STANDARD...)
    ENV_TYPE = __check_environment_type(envs)
    
    # Apply Wrappers
    envs = __apply_wrappers(envs, env_type=ENV_TYPE)
    
    return envs


# Create Single Test Environment
def create_test_environment(name, render_mode='rgb_array') -> gym.Env:
  return __create_test_environment(name, render_mode)

def __create_test_environment(name, render_mode='rgb_array'):
    
    # Build the Environment
    try: env = gym.make(name, render_mode=render_mode)
    except:
      
      # Not-Standard Render Mode 
      try: env = gym.make(name, render=True)
      except: env = gym.make(name)

    # Apply Wrappers
    env = __apply_wrappers(env, env_type = __check_environment_type(env))
    
    # Apply Record Video Wrapper -> Trigger Each Episode
    env = gym.wrappers.RecordVideo(env, video_folder=VIDEO_FOLDER, episode_trigger=lambda x: x % 1 == 0)
    
    return env

def __apply_wrappers(env, env_type):
      
  # Apply Specific Wrappers form GOAl Environments
  if env_type == GOAL_ENV:
        
        print('/n/nGOAL Environment/n/n')
        
        # Filter Out the Achieved Goal
        env = gym.wrappers.FilterObservation(env, ['observation', 'desired_goal'])
        
        # Flatten the Dictionary in a Flatten Array
        env = gym.wrappers.FlattenObservation(env)
  
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
  
  # OrderedDict Type
  from collections import OrderedDict
  
  # Check if is a Goal-Environment
  if type(env.reset()[0]) is OrderedDict and 'observation' in env.reset()[0]: return GOAL_ENV
  else: return STANDARD_ENV

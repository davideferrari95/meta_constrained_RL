import os, numpy as np
from typing import Optional

# Import Utils
from utils.Utils import VIDEO_FOLDER, VIOLATIONS_FOLDER, TEST_FOLDER
from utils.Utils import video_rename

# Import Environments
import gym, safety_gym
import mujoco_py

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

    # Build the Environment
    else: env = gym.make(name, render_mode=render_mode)

    # Apply Wrappers
    if   environment_type == 'test'      and record_video: env = gym.wrappers.RecordVideo(env, video_folder=TEST_FOLDER,       episode_trigger=lambda x: x % env_epochs == 0, name_prefix='test_')
    elif environment_type == 'violation' and record_video: env = gym.wrappers.RecordVideo(env, video_folder=VIOLATIONS_FOLDER, episode_trigger=lambda x: x % env_epochs == 0, name_prefix='new_')
    elif apply_wrappers: env = __apply_wrappers(env, record_video, record_epochs, folder=VIDEO_FOLDER)

    # TODO: Apply Seed -> In env.reset() ?
    # env.seed(seed)

    return env

  return make_env

def __make_custom_env(name, config:dict, render_mode='rgb_array') -> gym.Env: #():

  """ Custom environments used in the paper (taken from the official implementation) """

  from envs.DefaultEnvironment import static_config, dynamic_config
  from safety_gym.envs.engine import Engine
  from gym import register

  # Build Static / Dynamic Custom Environment
  if   "static"  in name: name, config = 'StaticEnv-v0',  static_config
  elif "dynamic" in name: name, config = 'DynamicEnv-v0', dynamic_config

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

def __apply_wrappers(env, record_video, record_epochs, folder) -> gym.Env: #():

  """ Apply Gym Wrappers """

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

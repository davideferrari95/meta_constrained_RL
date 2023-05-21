import os
from typing import Optional

# Import Utils
from utils.Utils import VIDEO_FOLDER, VIOLATIONS_FOLDER, TEST_FOLDER
from utils.Utils import video_rename

# Import Environments
import gym, safety_gym

# Import Custom `gym.make` Function
from envs.gym.Make import make

# Import Custom Gym Wrappers
from envs.gym.wrappers.RecordVideo import RecordVideo
from envs.gym.wrappers.RecordEpisodeStatistics import RecordEpisodeStatistics

# Import Vectorized Environment
from gym.vector import SyncVectorEnv

# Create Vectorized Environments
def create_vectorized_environment(name:str, config:Optional[dict]=None, env_num:int=10, record_video:bool=True,
                                  record_epochs:int=100, record_first_epoch:bool=False,
                                  render_mode='rgb_array', apply_wrappers:bool=True,
                                  environment_type:Optional[str]=None, env_epochs:int=1) -> SyncVectorEnv:

    """ Create Vectorized Gym Environment """

    def make_env(record_video):
        def thunk(): return create_environment(name, config, record_video, record_epochs, record_first_epoch, 
                                               render_mode, apply_wrappers, environment_type, env_epochs)
        return thunk

    # Create Vectorized Environment (Record Video Only for First Environment)
    return SyncVectorEnv([make_env(record_video if i == 0 else False) for i in range(env_num)])

# Create Single Environment
def create_environment(name:str, config:Optional[dict]=None, record_video:bool=True,
                       record_epochs:int=100, record_first_epoch:bool=False,
                       render_mode='rgb_array', apply_wrappers:bool=True,
                       environment_type:Optional[str]=None, env_epochs:int=1) -> gym.Env:

    """ Create Gym Environment """

    # Custom Environment Creation
    if ('custom' in name) or (config is not None): env = __make_custom_env(name, config, render_mode)

    # Build the Environment
    else: env = gym.make(name, render_mode=render_mode)

    # Apply Wrappers
    if   environment_type == 'test'      and record_video: env = RecordVideo(env, video_folder=TEST_FOLDER,       episode_trigger=lambda x: x % env_epochs == 0, name_prefix='test_')
    elif environment_type == 'violation' and record_video: env = RecordVideo(env, video_folder=VIOLATIONS_FOLDER, episode_trigger=lambda x: x % env_epochs == 0, name_prefix='new_')
    elif apply_wrappers: env = __apply_wrappers(env, record_video, record_epochs, record_first_epoch, folder=VIDEO_FOLDER)

    return env

def __make_custom_env(name, config:Optional[dict]=None, render_mode='rgb_array') -> gym.Env:

    """ Custom Environments used in the Paper (Official Implementation) """

    from gym import register
    from gym.envs.registration import registry

    # If Name is not in Pre-Configured Environments and Config is None
    if config is None: raise Exception(f"{name} Environment Not Implemented")

    # Use Given Config | Remove 'custom' from name, Capitalize and add '-v0'
    name = (''.join(name)).replace('custom','') + '-v0'

    # Check if Environment is Already Registered
    if name not in registry:

        register(
            id=name,
            entry_point="safety_gym.envs.safety_mujoco:Engine",
            max_episode_steps=1000,
            kwargs={"config": config},
        )

    # Custom `make` Function
    return make(name, render_mode=render_mode)

def __apply_wrappers(env, record_video, record_epochs, record_first_epoch, folder) -> gym.Env:

    """ Apply Custom Gym Wrappers """

    # Define Record Episode Trigger Function
    if record_first_epoch: episode_trigger = lambda x: x % record_epochs == 0
    else: episode_trigger = lambda x: x % record_epochs == 0 and x != 0

    # FIX: MoviePy Log Removed
    # Record Environment Videos in the specified folder, trigger specifies which episode to record and which to ignore (1 in record_epochs)
    if record_video: env = RecordVideo(env, video_folder=folder, episode_trigger=episode_trigger)

    # Keep Track of the Reward the Agent Obtain and Save them into a Property
    env = RecordEpisodeStatistics(env)

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
        else: number = None

        if filename.startswith('test_'): video_rename(TEST_FOLDER, filename, f'{prefix}-test-episode-{number}')

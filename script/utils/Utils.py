import os, io, random, omegaconf
import torch
import numpy as np
from scipy import signal
from termcolor import colored
from typing import Optional, Tuple

# Utils
AUTO = 'auto'

# Project Folder (ROOT Project Location)
FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))

# Check Video Folder and Return /Videos/Trial_{n}
def check_video_folder(folder):

    # Check Existing Video Folders
    n = 0
    while True:
        if not os.path.exists(f'{folder}/data/videos/Trial_{n}'): break
        else: n += 1

    return f'{folder}/data/videos/Trial_{n}'

# Define Video and Violations Folder
VIDEO_FOLDER = check_video_folder(FOLDER)
VIOLATIONS_FOLDER = f'{VIDEO_FOLDER}/Violations'
TEST_FOLDER = f'{VIDEO_FOLDER}/Test'

def set_hydra_absolute_path():

    import yaml
    hydra_config_file = os.path.join(FOLDER, 'config/config.yaml')

    # Load Hydra `config.yaml` File
    with open(hydra_config_file, 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)

    # Edit Hydra Run Directory
    yaml_data['hydra']['run']['dir'] = os.path.join(FOLDER, r'data/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}')

    # Write Hydra `config.yaml` File
    with open(hydra_config_file, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

# Print and Save Arguments
def print_arguments(cfg, term_print = True, save_file = False):

    if term_print: print(colored('\n\nArguments:\n', 'green'))

    if save_file:

        # Create Directory
        if not os.path.exists(VIDEO_FOLDER): os.mkdir(VIDEO_FOLDER)

        # Create File Info.txt
        file = open(f'{VIDEO_FOLDER}/# Info.txt', 'w')

        file.write('Arguments:\n')

    else: file = None

    # Recursive Print Function
    _recursive_print(cfg, file, space='   ', term_print=term_print)

    # Close Save File
    if file is not None: file.close()

def _recursive_print(cfg, file: Optional[io.TextIOWrapper]=None, space='   ', term_print=True):

    new_line = False

    for arg in cfg:

        # Print Argument Group Names
        if type(cfg[arg]) is omegaconf.dictconfig.DictConfig:

            # Terminal Print Argument Group
            if term_print: print(f"\n{colored(f'{space}{arg}:', 'yellow', attrs=['bold'])}\n")

            # Save File Info Arguments
            if file is not None: file.write(f'\n{space}{arg}:\n\n')

            _recursive_print(cfg[arg], file=file, space=f'{space}   ', term_print=term_print)

            # New Line Required for New Group
            new_line = True

        # Print Arguments
        else:

            # Change Arg Name if _target_ Class
            arg_name = 'class:' if arg == '_target_' else f'{arg}:'

            # Print Arguments
            if term_print: print (('\n' if new_line else ''), colored((f'{space[:-1]}{arg_name:25}'),
                                  ('red' if arg == '_target_' else 'white'), attrs=['bold']),
                                  f'{cfg[arg]}', '\n' if arg == '_target_' else '')

            # Save into File
            if file is not None:
                file.write(('\n' if new_line else ''))
                file.write(f'{space}{arg_name:25}{cfg[arg]}\n')

        if arg == 'utilities_params':

            # Compute Usage Booleans
            use_costs = bool(cfg['cost_params']['fixed_cost_penalty'] or cfg['cost_params']['target_cost'] or cfg['cost_params']['cost_limit'])
            learn_alpha = True if cfg['entropy_params']['alpha'] == AUTO else False
            learn_beta  = True if use_costs and cfg['cost_params']['fixed_cost_penalty'] is None else False

            # Print Arguments
            if term_print:
                print(colored(f'\n{space}   {"use_costs:":25}', 'white', attrs=['bold']), f'{use_costs}')
                print(colored(f'{space}   {"learn_alpha:":25}', 'white', attrs=['bold']), f'{learn_alpha}')
                print(colored(f'{space}   {"learn_beta:":25}',  'white', attrs=['bold']), f'{learn_beta}')

            # Save Info Arguments
            if file is not None:
                file.write(f'{space}   {"use_costs:":25}{use_costs}\n')
                file.write(f'{space}   {"learn_alpha:":25}{learn_alpha}\n')
                file.write(f'{space}   {"learn_beta:":25}{learn_beta}\n')

def check_spells_error(cfg):

    for arg in cfg:

        # Recursive Check None if New Group
        if type(cfg[arg]) is omegaconf.dictconfig.DictConfig: check_spells_error(cfg[arg])

        # Check if 'None' or 'Null'
        elif type(cfg[arg]) is str and (cfg[arg]).lower() in ['none', 'null']: cfg[arg] = None

        # Check if 'AUTO' or 'auto'
        elif type(cfg[arg]) is str and (cfg[arg]).lower() in ['auto']: cfg[arg] = AUTO

    return cfg

def set_seed_everywhere(seed):

    """ Apply Seeding Everywhere """

    print('\n\n')

    if seed == -1: 

        # Generating Random Seed
        seed = np.random.randint(0,2**32)
        print(f"Seed Must be Provided, Generating Random Seed: {seed}")

    # Set Manual Seed in PyTorch-Lightning 
    import pytorch_lightning as pl
    pl.seed_everything(seed)

    # Set Manual Seed in PyTorch Networks
    torch.manual_seed(seed)

    # Set Manual Seed in CUDA 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set Manual Seed in Numpy, Random
    np.random.seed(seed)
    random.seed(seed)

    return seed

class CostMonitor():

    ''' Class that Monitors the safety-gym Environment Violation, Cost and Reward '''

    def __init__(self):

        # Violations, Cost, Reward
        self.hazards_violation, self.vases_violation = 0.0, 0.0
        self.episode_cost = 0.0
        self.robot_stuck = 0.0

    def compute_cost(self, info_dict):

        # Get Cumulative Cost | Update Episode Cost 
        cost = info_dict.get('cost', 0)
        self.episode_cost += cost

        # Single Cost Components
        cost_hazards = info_dict.get('cost_hazards', 0)
        cost_vases = info_dict.get('cost_vases', 0)

        # Episode Cost Components
        self.hazards_violation += cost_hazards
        self.vases_violation += cost_vases

        return cost

    @property
    def get_episode_cost(self):
        return self.episode_cost

    @property
    def get_hazards_violation(self):
        return self.hazards_violation

    @property
    def get_vases_violation(self):
        return self.vases_violation

    @property
    def get_robot_stuck(self):
        return self.robot_stuck

def video_rename(folder:str, old_name:str, new_name:str):

    # Check if .mp4 or .json
    if old_name.endswith('.mp4'): new_name = f'{new_name}.mp4'
    elif old_name.endswith('.json'): new_name = f'{new_name}.meta.json'

    # Rename Files
    os.rename(os.path.join(folder, old_name), os.path.join(folder, new_name))

def combined_shape(length, shape=None) -> Tuple:

    """ Combine Length and Shape of an Array """

    if shape is None: return (length,)

    # Return Combined Shape of an Array
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumulative_sum(x, discount):

    """
    Compute Discounted Cumulative Sums of Vectors.
        input: vector x, [x0, x1, x2]
        output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
    """

    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def statistics_scalar(x, with_min_and_max=False) -> Tuple:

    """
    Get mean/std and optional min/max of scalar x.

    Args:
        x: An array containing samples of the scalar to produce statistics for.
        with_min_and_max (bool): If true, return min and max of x in addition to mean and std.
    """

    # Convert to Float32 Numpy Array
    x = np.array(x, dtype=np.float32)

    # Compute Mean
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n

    # Compute Standard Deviation
    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)

    if with_min_and_max:

        # Compute Min and Max
        global_min = np.min(x) if len(x) > 0 else np.inf
        global_max = np.max(x) if len(x) > 0 else -np.inf

        # Return Mean, Std, Min, Max
        return mean, std, global_min, global_max

    # Return Mean, Std
    return mean, std

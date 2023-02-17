import os, random, omegaconf, sys
import torch
import numpy as np
from termcolor import colored

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

VIDEO_FOLDER = check_video_folder(FOLDER)

# Print and Save Arguments
def print_arguments(cfg, term_print = True, save_file = False):
    
    if term_print: print(colored('\n\nArguments:\n', 'green'))
    
    if save_file:
        
        # Create Directory
        if not os.path.exists(VIDEO_FOLDER): os.mkdir(VIDEO_FOLDER)
        
        # Create File Info.txt
        file = open(f'{VIDEO_FOLDER}/# Info.txt', 'w')
    
        file.write('Arguments:\n')
    
    # Recursive Print Function
    _recursive_print(cfg, (file if save_file else None), space='   ', term_print=term_print, save_file=save_file)    
    
    # Close Save File
    if save_file: file.close()

def _recursive_print(cfg, file=None, space='   ', term_print=True, save_file=False):
    
    new_line = False
    
    for arg in cfg:
        
        # Print Argument Group Names
        if type(cfg[arg]) is omegaconf.dictconfig.DictConfig:
            
            # Terminal Print Argument Group
            if term_print: print(f"\n{colored(f'{space}{arg}:', 'yellow', attrs=['bold'])}\n")
            
            # Save File Info Arguments
            if save_file: file.write(f'\n{space}{arg}:\n\n')
            
            _recursive_print(cfg[arg], file=file, space=f'{space}   ', term_print=term_print, save_file=save_file)
            
            # New Line Required for New Group
            new_line = True

        # Print Arguments
        else:
            
            # Change Arg Name if _target_ Class
            arg_name = 'class' if arg == '_target_' else arg
                
            # Get Tabulation Length
            length = (len('class') if arg == '_target_' else len(arg)) + len(space)
            tab = '' if length >= 24 else '   ' if length >= 22 else '\t  ' if length >= 14 else '\t\t  ' if length >= 8 else '\t\t\t  '
            
            # Print Arguments
            if term_print: print (('\n' if new_line else ''), colored((f'{space[:-1]}{arg_name}:'), 
                                  ('red' if arg == '_target_' else 'white'), attrs=['bold']),
                                  f'{tab}{cfg[arg]}', '\n' if arg == '_target_' else '')
            
            # Save into File
            if save_file:
                file.write(('\n' if new_line else ''))
                file.write(f'{space}{arg_name}:{tab}{cfg[arg]}\n')
        
        if arg == 'utilities_params':
            
            # Compute Usage Booleans
            use_costs = bool(cfg['cost_params']['fixed_cost_penalty'] or cfg['cost_params']['target_cost'] or cfg['cost_params']['cost_limit'])
            learn_alpha = True if cfg['entropy_params']['alpha'] == AUTO else False
            learn_beta  = True if use_costs and cfg['cost_params']['fixed_cost_penalty'] is None else False
        
            # Print Arguments
            if term_print:
                print(colored(f'\n{space}   use_costs:', 'white', attrs=['bold']), f'\t  {use_costs}')
                print(colored(f'{space}   learn_alpha:', 'white', attrs=['bold']), f'\t  {learn_alpha}')
                print(colored(f'{space}   learn_beta:',  'white', attrs=['bold']), f'\t  {learn_beta}')
            
            # Save Info Arguments
            if save_file:
                file.write(f'{space}   use_costs:  \t  {use_costs}\n')
                file.write(f'{space}   learn_alpha:\t  {learn_alpha}\n')
                file.write(f'{space}   learn_beta: \t  {learn_beta}\n')

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
    
    # Apply Manual Seed to Torch Networks
    torch.manual_seed(seed)
    
    # Apply Manual Seed in Cuda 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Apply Seeding in Numpy and Random
    np.random.seed(seed)
    random.seed(seed)

def print_float_array(text, list_of_floats, decimal_number=4):
    
    list_of_float = [((f'{item:.0f}') if item * 10 % 10 == 0 
                    else (f'{item:.{decimal_number}f}')) 
                    for item in list_of_floats]
    
    print(f'{text} [', end='')
    print(*list_of_float, sep=", ", end=']\n')
    
def is_between(value, min_value, max_value):
    
    ''' Is Between Two Numbers '''
    
    assert min_value < max_value, f'Min Value > Max Value'
    
    return min_value <= value <= max_value

def is_between_180(value, min_angle, max_angle, extern_angle=False):
    
    ''' Is Between Two Angles in a +- 180 Circumference '''
    
    if not extern_angle: return is_between(value, min_angle, max_angle)
    else: return is_between(value, -180, min_angle) or is_between(value, max_angle, 180)

def get_index(vec1, vec2):

    ''' Check if Vector 1 is Contained in Vector 2 and Return the Index '''
    
    assert len(vec1) <= len(vec2), f'Vector 1 Must be <= Vector 2'
    
    # If Array are Equal Return 0
    if np.array_equal(vec1, vec2[:len(vec1)]): return 0
    
    # Slice the Bigger Array
    for i in range(1, len(vec2) - len(vec1) + 1):
        
        # Return the Slicing Index if Array are Equal
        if np.array_equal(vec1, vec2[i:i+len(vec1)]): return i
    
    return None

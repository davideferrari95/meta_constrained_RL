import os
from termcolor import colored
import omegaconf

# Utils
AUTO = 'auto'

# Project Folder
FOLDER = f'{os.path.dirname(__file__)}/'


# Check Video Folder and Return /Videos/Trial_{n}
def check_video_folder(folder):
  
    # Check Existing Video Folders
    n = 0
    while True:
        if not os.path.exists(f'{folder}/Videos/Trial_{n}'): break
        else: n += 1
        
    return f'{folder}/Videos/Trial_{n}'

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
            tab = '' if length >= 24 else ' ' if length >= 23 else '\t  ' if length >= 14 else '\t\t  ' if length >= 8 else '\t\t\t  '
            
            # Print Arguments
            if term_print: print (('\n' if new_line else ''), colored((f'{space[:-1]}{arg_name}:'), 
                                  ('red' if arg == '_target_' else 'white'), attrs=['bold']),
                                  f'{tab}{cfg[arg]}', '\n' if arg == '_target_' else '')
            
            # Save into File
            if save_file:
                file.write(('\n' if new_line else ''))
                file.write(f'{space}{arg_name}:{tab}{cfg[arg]}\n')
        
        if arg == 'utilities_params':
            
            # Compute Cost Usage
            use_costs = bool(cfg['cost_params']['fixed_cost_penalty'] or cfg['cost_params']['target_cost'] or cfg['cost_params']['cost_limit'])
            
            # Print Arguments
            if term_print: print (colored(f'{space}   use_costs:', 'white', attrs=['bold']), f'\t  {use_costs}')
            
            # Save Info Arguments
            if save_file: file.write(f'{space}   use_costs:\t  {use_costs}\n')


def check_spells_error(cfg):
    
    for arg in cfg:

        # Recursive Check None if New Group
        if type(cfg[arg]) is omegaconf.dictconfig.DictConfig: check_spells_error(cfg[arg])

        # Check if 'None' or 'Null'
        elif type(cfg[arg]) is str and (cfg[arg]).lower() in ['none', 'null']: cfg[arg] = None

        # Check if 'AUTO' or 'auto'
        elif type(cfg[arg]) is str and (cfg[arg]).lower() in ['auto']: cfg[arg] = AUTO

    return cfg

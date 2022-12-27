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
def print_arguments(args, term_print = True, save = False):
    
    if term_print: print(colored('\n\nArguments:\n', 'green'))
    
    if save:
        
        # Create Directory
        if not os.path.exists(VIDEO_FOLDER): os.mkdir(VIDEO_FOLDER)
        
        # Create File Info.txt
        file = open(f'{VIDEO_FOLDER}/# Info.txt', 'w')
    
        file.write('Arguments:\n\n')

    for arg in vars(args):
        
        new_line = '\n' if arg in ['tau','init_alpha','cost_limit','record_video','fast_dev_run'] else ''
        
        # Print Arguments
        print_arg(arg, getattr(args, arg), term_print, new_line)
        
        # Get Tabulation Length
        tab = '' if len(arg) >= 18 else '\t' if len(arg) >= 9 else '\t\t' 
        
        # Save Info Arguments
        if save: file.write(f'   {arg}: {tab}{getattr(args, arg)}\n{new_line}')

    # Close Save File
    if save: file.close()

def print_arg(arg, value, term_print = True, new_line=''):
    
    # Print Arguments
    tab = '' if len(arg) >= 18 else '\t' if len(arg) >= 9 else '\t\t' 
    if term_print: print (colored(f'   {arg}: ', 'white', attrs=['bold']), f'{tab}{value}{new_line}')

def check_none(args):
    
    for arg in args:
        
        if type(args[arg]) is omegaconf.dictconfig.DictConfig:
            # print(f'recursive, {arg}')
            # print(args[arg])
            check_none(args[arg])

        else:
            # print(f'{arg}: {args[arg]}')
            # Check if an Arguments Contains 'None' as String
            if type(args[arg]) is str and (args[arg]).lower() in ['none', 'null']: args[arg] = None
    
    return args

import os
from termcolor import colored

# Video Folder
FOLDER = f'{os.path.dirname(__file__)}/'

# Default Environment
# ENV  = 'Safexp-PointGoal1-v0'
ENV  = 'Safexp-PointGoal2-v0'
# ENV  = 'Safexp-CarGoal2-v0'

def print_arguments(args):
    
    print(colored('\n\nArguments:\n', 'green'))
    
    for arg in vars(args):
        tab = '\t' if len(arg) > 11 else '\t\t'
        print (colored(f'   {arg}: ', 'white', attrs=['bold']), f'{tab}{getattr(args, arg)}')
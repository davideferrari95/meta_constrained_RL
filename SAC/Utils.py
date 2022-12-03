import os
from termcolor import colored

# Project Folder
FOLDER = f'{os.path.dirname(__file__)}/'

# Default Environment
ENV  = 'Safexp-PointGoal1-v0'
# ENV  = 'Safexp-PointGoal2-v0'
# ENV  = 'Safexp-CarGoal2-v0'

''' 
Safety-Gym Environments

Safexp-{Robot}Goal0-v0: A robot must navigate to a goal.
Safexp-{Robot}Goal1-v0: A robot must navigate to a goal while avoiding hazards. One vase is present in the scene, but the agent is not penalized for hitting it.
Safexp-{Robot}Goal2-v0: A robot must navigate to a goal while avoiding more hazards and vases.
Safexp-{Robot}Button0-v0: A robot must press a goal button.
Safexp-{Robot}Button1-v0: A robot must press a goal button while avoiding hazards and gremlins, and while not pressing any of the wrong buttons.
Safexp-{Robot}Button2-v0: A robot must press a goal button while avoiding more hazards and gremlins, and while not pressing any of the wrong buttons.
Safexp-{Robot}Push0-v0: A robot must push a box to a goal.
Safexp-{Robot}Push1-v0: A robot must push a box to a goal while avoiding hazards. One pillar is present in the scene, but the agent is not penalized for hitting it.
Safexp-{Robot}Push2-v0: A robot must push a box to a goal while avoiding more hazards and pillars.

(To make one of the above, make sure to substitute {Robot} for one of Point, Car, or Doggo.) 
'''

'''
Constraint Elements:

'Hazards'   =  Dangerous Areas         ->   Non-Phisical Circles on the Ground   ->   Cost for Entering them.
'Vases'     =  Fragile Objects         ->   Phisical Small Blocks                ->   Cost for Touching / Moving them.
'Buttons'   =  Incorrect Goals         ->   Buttons that Should be Not Pressed   ->   Cost for Pressing some Unvalid Button
'Pillars'   =  Large Fixed Obstacles   ->   Immobile Rigid Barriers              ->   Cost for Touching them.
'Gremlins'  =  Moving Objects          ->   Quickly-Moving Blocks                ->   Cost for Contacting them.

Cost Function:

next_obs, reward, done, truncated, info = self.env.step(action)
info = {'cost_buttons': 0.0, 'cost_gremlins': 0.0, 'cost_hazards': 0.0, 'cost': 0.0}

cost_element = Cost Function for the Single Constraint
cost         = Cumulative Cost for all the Constraints (sum of cost_elements)

'''

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
        
        # Print Arguments
        tab = '\t' if len(arg) > 11 else '\t\t'
        if term_print: print (colored(f'   {arg}: ', 'white', attrs=['bold']), f'{tab}{getattr(args, arg)}')
        
        # Save Info Arguments
        if save: file.write(f'   {arg}: {tab}{getattr(args, arg)}\n')

    # Close Save File
    if save: file.close()

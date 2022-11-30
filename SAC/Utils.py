from termcolor import colored

# Check Gym Version
def check_gym_version():
    
    import gym, packaging.version as version
    if version.parse(gym.__version__) >= version.parse('0.26'): return True, None
    else: return False, print(colored(f'\nWARNING! Old Version of gym: {gym.__version__}', 'red'))

GYM_NEW, _ = check_gym_version()

import argparse

# Import Algorithm
from SAC.SAC import SAC

# Import PyTorch Lightning
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from SAC.LightningCallbacks import PrintCallback

# Import Utilities
from SAC.Utils import ENV, FOLDER, print_arguments

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

if __name__ == '__main__':
    
    # Instantiate Parser
    parser = argparse.ArgumentParser()
    
    # Parse Arguments
    parser.add_argument('--env',                type=str,    default=ENV)
    parser.add_argument('--epochs',             type=int,    default=10_000)
    parser.add_argument('--samples_per_epoch',  type=int,    default=1_000)
    parser.add_argument('--alpha',              type=float,  default=0.002)
    parser.add_argument('--beta',               type=float,  default=0.002)
    parser.add_argument('--tau',                type=float,  default=0.1)
    parser.add_argument('--fast_dev_run',       type=bool,   default=False)
    args = parser.parse_args()

    # Display Arguments
    print_arguments(args)

    # Instantiate Algorithm
    algorithm = SAC(env_name=args.env, folder_name=FOLDER, samples_per_epoch=args.samples_per_epoch, 
                    alpha=args.alpha, beta=args.beta, tau=args.tau)
    
    # Create Trainer Module
    trainer = Trainer(
        
        # Devices
        devices="auto", 
        accelerator="auto",
        
        # Hyperparameters
        max_epochs  = args.epochs,
        
        # Additional Callbacks
        callbacks   = [PrintCallback(),
                       EarlyStopping(monitor='episode/Return', mode='max', patience=500, verbose=True),
                       ],
        
        # Custom TensorBoard Logger
        logger = pl_loggers.TensorBoardLogger(save_dir=f'{FOLDER}/Logs/'),
        
        # Developer Test Mode
        fast_dev_run = args.fast_dev_run

    )
    
    trainer.fit(algorithm)

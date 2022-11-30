import os

# Import Algorithm
from SAC.SAC import SAC
from SAC.Utils import check_gym_version

# Import PyTorch Lightning
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from SAC.LightningCallbacks import PrintCallback

# Hyperparameters
FOLDER = f'{os.path.dirname(__file__)}/'
EPOCHS = 10_000
FAST_DEV_RUN = False

# ENV = 'LunarLanderContinuous-v2'
# ENV  = 'Safexp-PointGoal1-v0'
# ENV  = 'Safexp-PointGoal0-v0'
# ENV  = 'Safexp-PointGoal2-v0'
ENV  = 'Safexp-CarGoal2-v0'

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

if __name__ == '__main__':
    
    # Instantiate Algorithm
    algorithm = SAC(env_name=ENV, folder_name=FOLDER, samples_per_epoch=1_000, alpha=0.002, tau=0.1)
    
    # Create Trainer Module
    trainer = Trainer(
        
        # Devices
        devices="auto", 
        accelerator="auto",
        
        # Hyperparameters
        max_epochs  = EPOCHS,
        
        # Additional Callbacks
        callbacks   = [PrintCallback(),
                       EarlyStopping(monitor='episode/Return', mode='max', patience=500, verbose=True),
                       ],
        
        # Custom TensorBoard Logger
        logger = pl_loggers.TensorBoardLogger(save_dir=f'{FOLDER}/Logs/'),
        
        # Developer Test Mode
        fast_dev_run = FAST_DEV_RUN

    )
    
    trainer.fit(algorithm)

import argparse

# Import Algorithm
from SAC.SAC import SAC

# Import PyTorch Lightning
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from SAC.LightningCallbacks import PrintCallback

# Import Utilities
from SAC.Utils import ENV, FOLDER, AUTO, print_arguments
from typing import Union, Optional

if __name__ == '__main__':
    
    # Instantiate Argument Parser
    parser = argparse.ArgumentParser()
    
    # Learning HyperParameters
    parser.add_argument('--env',                type=str,    default=ENV)
    parser.add_argument('--epochs',             type=int,    default=10_000)
    parser.add_argument('--samples_per_epoch',  type=int,    default=10_000)
    parser.add_argument('--patience',           type=int,    default=1_000)
    parser.add_argument('--tau',                type=float,  default=0.1)

    # Trainable HyperParameters
    parser.add_argument('--alpha',              default=AUTO)
    parser.add_argument('--target_alpha',       default=AUTO)
    parser.add_argument('--init_alpha',         default=None)
    # parser.add_argument('--beta',               default=AUTO)
    parser.add_argument('--beta',               default=0.05)
    parser.add_argument('--target_beta',        default=AUTO)
    parser.add_argument('--init_beta',          default=None)
    
    # Training Options
    parser.add_argument('--fast_dev_run',       action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    # Display Arguments
    print_arguments(args, term_print=True, save=False)

    # Instantiate Algorithm Model
    model = SAC(env_name = args.env, samples_per_epoch = args.samples_per_epoch, tau = args.tau, 
                alpha = args.alpha, target_alpha = args.target_alpha, init_alpha = args.init_alpha,
                beta  = args.beta,  target_beta  = args.target_beta,  init_beta  = args.init_beta)
    
    # Create Trainer Module
    trainer = Trainer(
        
        # Devices
        devices="auto", 
        accelerator="auto",
        
        # Hyperparameters
        max_epochs  = args.epochs,
        
        # Additional Callbacks
        callbacks   = [PrintCallback(),
                       EarlyStopping(monitor='episode/Return', mode='max', patience=args.patience, verbose=True),
                       ],
        
        # Custom TensorBoard Logger
        logger = pl_loggers.TensorBoardLogger(save_dir=f'{FOLDER}/Logs/'),
        
        # Developer Test Mode
        fast_dev_run = args.fast_dev_run

    )
        
    # Save Arguments
    print_arguments(args, term_print=False, save=True)
        
    # Start Traiing
    trainer.fit(model)

import argparse

# Import Algorithm
from SAC.SAC import SAC

# Import PyTorch Lightning
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from SAC.LightningCallbacks import PrintCallback

# Import Utilities
from SAC.Utils import ENV, FOLDER, print_arguments

if __name__ == '__main__':
    
    # Instantiate Parser
    parser = argparse.ArgumentParser()
    
    # Parse Arguments
    parser.add_argument('--env',                type=str,    default=ENV)
    parser.add_argument('--epochs',             type=int,    default=10_000)
    parser.add_argument('--samples_per_epoch',  type=int,    default=10_000)
    parser.add_argument('--alpha',              type=float,  default=0.002)
    parser.add_argument('--beta',               type=float,  default=0.01)
    parser.add_argument('--tau',                type=float,  default=0.1)
    parser.add_argument('--fast_dev_run',       action=argparse.BooleanOptionalAction, default=False)
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

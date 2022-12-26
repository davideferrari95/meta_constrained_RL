# Import Algorithm
from SAC.SAC import SAC

# Import PyTorch Lightning
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from SAC.LightningCallbacks import PrintCallback

# Import Utilities
from SAC.Utils import ENV, FOLDER, AUTO, print_arguments, check_none

# Import Hydra, Parameters Configuration File
import hydra
from config.config import Params

# Hydra Decorator to Load Configuration Files
@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg: Params):
    
    # Check for 'None' or 'Null' Strings -> None
    cfg = check_none(cfg)
    
    # Display Arguments
    # print_arguments(args, term_print=True, save=False)

    TP = cfg.training_params
    EP = cfg.entropy_params
    CP = cfg.cost_params
    UP = cfg.utilities_params

    # Instantiate Algorithm Model
    model = SAC(env_name = TP.env, record_video=(UP.record_video and not UP.fast_dev_run),
                samples_per_epoch = TP.samples_per_epoch if not UP.fast_dev_run else 1, tau = TP.tau, 
                alpha = EP.alpha, target_alpha = EP.target_alpha, init_alpha = EP.init_alpha,
                fixed_cost_penalty = CP.fixed_cost_penalty, cost_constraint = CP.cost_constraint, cost_limit = CP.cost_limit)

    # Instantiate Default Callbacks
    callbacks = [PrintCallback()]

    # Optional Callbacks
    if UP.early_stopping: callbacks.append(EarlyStopping(monitor='episode/Return', mode='max', patience=TP.patience, verbose=True))

    # Create Trainer Module
    trainer = Trainer(
        
        # Devices
        devices = "auto", 
        accelerator = "auto",
        
        # Hyperparameters
        max_epochs = TP.epochs,
        
        # Additional Callbacks
        callbacks = callbacks,
        
        # Custom TensorBoard Logger
        logger = pl_loggers.TensorBoardLogger(save_dir=f'{FOLDER}/Logs/'),
        
        # Developer Test Mode
        fast_dev_run = UP.fast_dev_run

    )
        
    # Save Arguments
    # print_arguments(args, term_print=False, save=(True and (args.record_video and not args.fast_dev_run)))

    # Start Training
    trainer.fit(model)


if __name__ == '__main__':

    main()

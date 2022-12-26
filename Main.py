import argparse

# Import Algorithm
from SAC.SAC import SAC

# Import PyTorch Lightning
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from SAC.LightningCallbacks import PrintCallback

# Import Utilities
from SAC.Utils import ENV, FOLDER, AUTO, print_arguments, check_none

# Import Hydra
import hydra
from hydra.core.config_store import ConfigStore

# Import Parameters Configuration File
from config.config import Params

# Instantiate Hydra Config Store
cs = ConfigStore.instance()
cs.store(name='param_config', node=Params)

# Hydra Decorator to Load Configuration Files
@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg: Params):

    # 'None' String
    # cfg = check_none(cfg)

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
    exit()
    
    # Instantiate Argument Parser
    parser = argparse.ArgumentParser()
    
    # Learning HyperParameters
    parser.add_argument('--env',                type=str,    default=ENV)
    parser.add_argument('--samples_per_epoch',  type=int,    default=10_000)
    parser.add_argument('--epochs',             type=int,    default=10_000)
    parser.add_argument('--patience',           type=int,    default=1_000)
    parser.add_argument('--tau',                type=float,  default=0.995)

    # Trainable HyperParameters
    parser.add_argument('--alpha',              default=AUTO)
    parser.add_argument('--target_alpha',       default=AUTO)
    parser.add_argument('--init_alpha',         default=None)
    parser.add_argument('--fixed_cost_penalty', default=None)
    parser.add_argument('--cost_constraint',    default=None)
    parser.add_argument('--cost_limit',         default=25)
    
    # Training Options (--no- to set False)
    parser.add_argument('--early_stopping',     action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--record_video',       action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fast_dev_run',       action=argparse.BooleanOptionalAction, default=False)

    # Parse Arguments and Check for 'None' String
    args = parser.parse_args()
    args = check_none(args)

    # TODO: Argparse Terminal Arguments Suggestion

    # Display Arguments
    print_arguments(args, term_print=True, save=False)

    # Instantiate Algorithm Model
    model = SAC(env_name = args.env, record_video=(args.record_video and not args.fast_dev_run),
                samples_per_epoch = args.samples_per_epoch if not args.fast_dev_run else 1, tau = args.tau, 
                alpha = args.alpha, target_alpha = args.target_alpha, init_alpha = args.init_alpha,
                fixed_cost_penalty = args.fixed_cost_penalty, cost_constraint = args.cost_constraint, cost_limit = args.cost_limit)
    
    # Instantiate Default Callbacks
    callbacks = [PrintCallback()]
    
    # Optional Callbacks
    if args.early_stopping: callbacks.append(EarlyStopping(monitor='episode/Return', mode='max', patience=args.patience, verbose=True))
    
    # Create Trainer Module
    trainer = Trainer(
        
        # Devices
        devices = "auto", 
        accelerator = "auto",
        
        # Hyperparameters
        max_epochs = args.epochs,
        
        # Additional Callbacks
        callbacks = callbacks,
        
        # Custom TensorBoard Logger
        logger = pl_loggers.TensorBoardLogger(save_dir=f'{FOLDER}/Logs/'),
        
        # Developer Test Mode
        fast_dev_run = args.fast_dev_run

    )
        
    # Save Arguments
    print_arguments(args, term_print=False, save=(True and (args.record_video and not args.fast_dev_run)))

    # Start Training
    trainer.fit(model)

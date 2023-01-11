# Import WCSAC Algorithm
from SAC.SAC import WCSAC

# Import PyTorch Lightning
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, DeviceStatsMonitor, ModelCheckpoint
from SAC.LightningCallbacks import PrintCallback, OverrideEpochStepCallback

# Import Utilities
from SAC.Utils import FOLDER, AUTO, print_arguments, check_spells_error

# Import Hydra, Parameters Configuration File
import hydra
from config.config import Params

# Hydra Decorator to Load Configuration Files
@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg: Params):
    
    # Check for 'None' or 'Null' Strings -> None
    cfg = check_spells_error(cfg)
    
    # Display Arguments
    print_arguments(cfg, term_print=True, save_file=False)

    # Training and Utilities Parameters
    TP = cfg.training_params
    UP = cfg.utilities_params

    # Instantiate Algorithm Model
    model = hydra.utils.instantiate(cfg.agent, record_video=(UP.record_video and not UP.fast_dev_run),
                                    samples_per_epoch = TP.samples_per_epoch if not UP.fast_dev_run else 1)

    # Instantiate Default Callbacks
    callbacks = [PrintCallback(), OverrideEpochStepCallback(), DeviceStatsMonitor()]

    # Model Checkpoint Callback
    # callbacks.append(ModelCheckpoint())
    
    # Optional Callbacks
    if UP.early_stopping: callbacks.append(EarlyStopping(monitor='episode/Return', mode='max', patience=TP.patience, verbose=True))

    # Create Trainer Module
    trainer = Trainer(
        
        # Devices
        devices = AUTO, 
        accelerator = AUTO,
        
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
    print_arguments(cfg, term_print=False, save_file=(True and (UP.record_video and not UP.fast_dev_run)))

    # Start Training
    trainer.fit(model)


if __name__ == '__main__':

    main()

# Import WCSAC Algorithm
from SAC.SAC import WCSACP

# Import PyTorch Lightning
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler
from pytorch_lightning.callbacks import EarlyStopping, DeviceStatsMonitor, ModelCheckpoint
from SAC.LightningCallbacks import PrintCallback, OverrideEpochStepCallback
from SAC.Utils import set_seed_everywhere

# Import Utilities
from SAC.Utils import FOLDER, AUTO, print_arguments, check_spells_error

# Import Parent Folders
import sys
sys.path.append(FOLDER)

# Import Hydra and Parameters Configuration File
import hydra
from config.config import Params

# Set Hydra Full Log Error
import os
os.environ['HYDRA_FULL_ERROR'] = '1'

# Hydra Decorator to Load Configuration Files
@hydra.main(config_path=f'{FOLDER}/config', config_name='config', version_base=None)
def main(cfg: Params):
    
    # Check for 'None' or 'Null' Strings -> None
    cfg = check_spells_error(cfg)
    
    # Display Arguments
    print_arguments(cfg, term_print=True, save_file=False)

    # Training and Utilities Parameters
    TP = cfg.training_params
    UP = cfg.utilities_params
    
    # Add PyTorch Lightning Seeding
    seed = set_seed_everywhere(cfg.training_params.seed)

    # Instantiate Algorithm Model
    model = hydra.utils.instantiate(cfg.agent, seed=seed, record_video=(UP.record_video and not UP.fast_dev_run),
                                    samples_per_epoch = TP.samples_per_epoch if not UP.fast_dev_run else 1)

    # Instantiate Default Callbacks
    callbacks = [PrintCallback()]

    # Device Stats Monitoring
    # callbacks.append(DeviceStatsMonitor())

    # Model Checkpoint Callback
    # callbacks.append(ModelCheckpoint())
    
    # Override Epochs Callback
    # callbacks.append(OverrideEpochStepCallback())

    # Optional Callbacks
    if UP.early_stopping: callbacks.append(EarlyStopping(monitor='episode/Return', mode='max', patience=TP.patience, verbose=True))

    # Python Profiler: Summary of All the Calls Made During Training
    # profiler = AdvancedProfiler()
    profiler = SimpleProfiler() if UP.use_profiler else None

    # Create Trainer Module
    trainer = Trainer(
        
        # Devices
        devices = AUTO, 
        accelerator = AUTO,
        
        # Hyperparameters
        max_epochs = TP.epochs,
        
        # Additional Callbacks
        callbacks = callbacks,

        # Use Python Profiler
        profiler = profiler,
        
        # Custom TensorBoard Logger
        logger = pl_loggers.TensorBoardLogger(save_dir=f'{FOLDER}/data/logs/'),
        
        # Developer Test Mode
        fast_dev_run = UP.fast_dev_run

    )
        
    # Save Arguments
    print_arguments(cfg, term_print=False, save_file=(True and (UP.record_video and not UP.fast_dev_run)))

    # Start Training
    trainer.fit(model)


if __name__ == '__main__':

    main()

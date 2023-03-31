# Import WCSAC Algorithm
from SAC.SAC import WCSACP

# Import PyTorch Lightning
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler
from pytorch_lightning.callbacks import EarlyStopping, DeviceStatsMonitor, ModelCheckpoint
from SAC.LightningCallbacks import PrintCallback, TestCallback, OverrideEpochStepCallback
from SAC.Utils import set_seed_everywhere, set_hydra_absolute_path

# Import Utilities
from SAC.Utils import FOLDER, AUTO, print_arguments, check_spells_error

# Import Parent Folders
import sys
sys.path.append(FOLDER)

# Import Hydra and Parameters Configuration File
import hydra
from config.config import Params, EnvironmentParams, TrainingParams

# Set Hydra Absolute FilePath in `config.yaml`
set_hydra_absolute_path()

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

    # Environment and Training Parameters
    EP: EnvironmentParams = cfg.environment_params
    TP: TrainingParams    = cfg.training_params

    # Add PyTorch Lightning Seeding
    seed = set_seed_everywhere(cfg.training_params.seed)

    # Set Test Episode Number to 0 if Fast-Dev-Run of No-Record-Video
    if (not TP.record_video or TP.fast_dev_run): cfg.agent.environment_config.test_episode_number = 0

    # Instantiate Algorithm Model
    model = hydra.utils.instantiate(cfg.agent, seed=seed, record_video=(TP.record_video and not TP.fast_dev_run),
                                    samples_per_epoch = TP.samples_per_epoch if not TP.fast_dev_run else 1)

    # Instantiate Default Callbacks
    callbacks = [PrintCallback()]

    # Device Stats Monitoring
    # callbacks.append(DeviceStatsMonitor())

    # Model Checkpoint Callback
    # callbacks.append(ModelCheckpoint())

    # Override Epochs Callback
    # callbacks.append(OverrideEpochStepCallback())

    # Optional Callbacks
    if TP.early_stopping: callbacks.append(EarlyStopping(monitor='episode/Return', mode='max', patience=TP.patience, verbose=True))

    # Test Callback
    if EP.test_environment: callbacks.append(TestCallback())

    # Python Profiler: Summary of All the Calls Made During Training
    # profiler = AdvancedProfiler()
    profiler = SimpleProfiler() if TP.use_profiler else None

    # Create Trainer Module
    trainer = Trainer(

        # Devices
        devices = AUTO,
        accelerator = AUTO,

        # Hyperparameters
        min_epochs = TP.min_epochs,
        max_epochs = TP.max_epochs,

        # Additional Callbacks
        callbacks = callbacks,

        # Use Python Profiler
        profiler = profiler,

        # Custom TensorBoard Logger
        logger = pl_loggers.TensorBoardLogger(save_dir=f'{FOLDER}/data/logs/'),

        # Developer Test Mode
        fast_dev_run = TP.fast_dev_run

    )

    # Save Arguments
    print_arguments(cfg, term_print=False, save_file=(True and (TP.record_video and not TP.fast_dev_run)))

    # Start Training
    trainer.fit(model)

if __name__ == '__main__':

    main()

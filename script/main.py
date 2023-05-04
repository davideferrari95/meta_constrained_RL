# Import WCSAC Algorithm
from algos.PPO import PPO

# Import PyTorch Lightning
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler
from pytorch_lightning.callbacks import EarlyStopping, DeviceStatsMonitor, ModelCheckpoint
from utils.LightningCallbacks import PrintCallback, TestCallback
from utils.Utils import set_seed_everywhere, set_hydra_absolute_path

# Import Utilities
from utils.Utils import FOLDER, AUTO, print_arguments, check_spells_error
import sys, os, logging, torch

# Import Parent Folders
sys.path.append(FOLDER)

# Import Hydra and Parameters Configuration File
import hydra
from config.config import Params, EnvironmentParams, TrainingParams

# Set Hydra Absolute FilePath in `config.yaml`
set_hydra_absolute_path()

# Set Hydra Full Log Error
os.environ['HYDRA_FULL_ERROR'] = '1'

# Ignore Torch Compiler INFO
logging.getLogger('torch._dynamo').setLevel(logging.ERROR)
logging.getLogger('torch._inductor').setLevel(logging.ERROR)

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
    seed = set_seed_everywhere(cfg.environment_params.seed)

    # Set Test Episode Number to 0 if Fast-Dev-Run of No-Record-Video
    if (not TP.record_video or TP.fast_dev_run): cfg.agent.environment_config.test_episode_number = 0

    # Instantiate Algorithm Model
    model = hydra.utils.instantiate(cfg.agent, seed=seed, record_video=(TP.record_video and not TP.fast_dev_run))

    # Instantiate Default Callbacks
    callbacks = [PrintCallback()]

    # Device Stats Monitoring
    # callbacks.append(DeviceStatsMonitor())

    # Model Checkpoint Callback
    # callbacks.append(ModelCheckpoint())

    # Optional Callbacks
    if TP.early_stopping: callbacks.append(EarlyStopping(monitor='kl_divergence', mode='min', patience=TP.patience, verbose=True, check_on_train_epoch_end=True))

    # Test Callback
    if EP.test_environment: callbacks.append(TestCallback())

    # Python Profiler: Summary of All the Calls Made During Training
    profiler = AdvancedProfiler() if TP.profiler == 'advanced' else SimpleProfiler()
    profiler = profiler if TP.use_profiler else None

    # Create Trainer Module
    trainer = Trainer(

        # Devices
        # devices = 'cpu',
        accelerator = 'cpu',
        # devices = AUTO,
        # accelerator = AUTO,
        # strategy = AUTO,
        precision = TP.precision,

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

    # Model Compilation
    compiled_model = torch.compile(model, mode=TP.compilation_mode) if TP.torch_compilation else model

    # Start Training
    trainer.fit(compiled_model)

if __name__ == '__main__':

    main()

from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from termcolor import colored

# Print Info Callback
class PrintCallback(Callback):
    
    # On Start Training
    def on_train_start(self, trainer, pl_module):
        print(colored('\n\nStart Training Process\n\n','yellow'))
    
    # On End Training
    def on_train_end(self, trainer, pl_module):
        print(colored('\n\nTraining Done\n\n','yellow'))

# Use Epochs instead of Steps in TensorBoard Log
class OverrideEpochStepCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_training_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log("step", trainer.current_epoch)

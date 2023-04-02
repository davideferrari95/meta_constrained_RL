from SAC.Environment import rename_test_episodes

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

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log("epochs", trainer.current_epoch)

class TestCallback(Callback):

    # On End Training
    def on_train_end(self, trainer, pl_module):

        # Return if No Test Episode
        if pl_module.EC.test_episode_number == 0: return

        print(colored('Reproducing Some Test Episodes...\n\n', 'yellow'))

        if pl_module.EC.test_unconstrained:

            # Play a Bunch of Un-Constrained Test Episodes
            pl_module.play_test_episodes(test_constrained = False)

            # Rename Unconstrained Episodes
            rename_test_episodes(prefix='unconstrained')

        if pl_module.EC.test_constrained:

            # Play a Bunch of Constrained Test Episodes
            pl_module.play_test_episodes(test_constrained = True)

            # Rename Constrained Episodes
            rename_test_episodes(prefix='constrained')

        print(colored('\n\n\nTest Done\n\n', 'yellow'))

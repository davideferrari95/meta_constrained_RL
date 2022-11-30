from pytorch_lightning.callbacks import Callback
from termcolor import colored

# Print Info Callback
class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print(colored('\n\nStart Training Process\n\n','yellow'))
    def on_train_end(self, trainer, pl_module):
        print(colored('\n\nTraining Done\n\n','yellow'))
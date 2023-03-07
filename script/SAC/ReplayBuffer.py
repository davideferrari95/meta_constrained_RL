import random
from collections import deque

# Import Torch IterableDataset
from torch.utils.data.dataset import IterableDataset

# Replay Buffer
class ReplayBuffer():

    # Capacity: Maximum Number of Observation the Buffer Holds
    def __init__(self, capacity):

        # Deque Data Structure will Automatically Delete the Oldest Entry to Make Space for the New One
        self.buffer = deque(maxlen=int(capacity))

    def __len__(self):

        # Return the Number of Elements in the Object
        return len(self.buffer)

    def append(self, experience):

        # Store New Experience in the Buffer
        self.buffer.append(experience)

    def sample(self, batch_size):

        # Get a Batch of Observation from the Buffer and Return them to the User
        return random.sample(self.buffer, batch_size)

# Lightning Iterable DataSet
class RLDataset(IterableDataset):

    ''' Must create a dataset that PyTorch-Lightning can use '''

    def __init__(self, buffer, sample_size=600):

        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):

        # Define what Happen when we Process one by one this Object
        for experience in self.buffer.sample(self.sample_size):

            ''' yield will return the experience in the position number
            0 and then wait for PyTorch to request the next item'''
            yield experience

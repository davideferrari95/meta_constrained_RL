import random
from collections import deque

# Import Torch IterableDataset
from torch.utils.data.dataset import IterableDataset

# Replay Buffer
class ReplayBuffer():
    
    # capacity: maximum number of observation the buffer holds
    def __init__(self, capacity):
        
        # deque data structure will automatically delete the oldest entry to make space for the new one
        self.buffer = deque(maxlen=capacity)
    
    def __len__(self):
        
        # Retunr the number of elements in the object
        return len(self.buffer)
    
    def append(self, experience):
    
        # Store new experience in the buffer
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        
        # Get a batch of observation from the buffer and return them to the user
        return random.sample(self.buffer, batch_size)


# Lightning Iterable DataSet
class RLDataset(IterableDataset):
    
  
    # Must create a dataset that PyTorch-Lightning can use
    
    def __init__(self, buffer, sample_size=600):
        
        self.buffer = buffer
        self.sample_size = sample_size
        
    def __iter__(self):
        
        # Define what happen when we process one by one this object
        for experience in self.buffer.sample(self.sample_size):
            
            ''' yield will return the experience in the position number 
            0 and then wait for pythorch to request the next item'''
            yield experience
            
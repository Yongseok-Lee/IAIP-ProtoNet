import numpy as np
import torch
from torch.utils.data.sampler import Sampler

# Define the prototypical batch sampler class
class PrototypicalBatchSampler(Sampler):
  '''
  Yield a batch of sample indices per episode.
  Indices are calculated by keeping in account num_classes and num_samples.
  In fact at every episode the batch indices will refer to num_support + num_query samples
  for num_classes random classes.

  __len__ returns the number of episodes per epoch. (same as self.episodes)

  '''
  def __init__(self, targets: int, episodes: int, num_classes: int, num_samples: int):
    '''
    Initialize the PrototypicalBatchSampler object.

    Args:
      targets     (int): An iterable containing all the targets for the current dataset samples.
                         Indices will be infered from this iterable.
      episodes    (int): Number of episodes per epoch.
      num_classes (int): Number of random classes per episode.
      num_samples (int): Number of samples per class per episode. (support + query)

    '''
    super().__init__(data_source = None)
    self.targets     = targets
    self.episodes    = episodes
    self.num_classes = num_classes
    self.num_samples = num_samples

    self.all_classes, self.counts = np.unique(self.targets, return_counts = True)
    self.all_classes   = torch.LongTensor(self.all_classes)
    self.total_classes = len(self.all_classes)

    # Create a matrix 'indices' of dim = total_classes * max(num of elements per class) filled with NaNs
    self.indices = torch.Tensor(np.nan * np.empty((self.total_classes, max(self.counts)), dtype = int))

    # For every class in 'targets', fill the relative row with 'indices' samples belonging to the class
    # in num_samples_per_class we store the number of samples per class/row
    self.num_samples_per_class = torch.zeros_like(self.all_classes)

    for i, label in enumerate(self.targets):
      class_index = np.argwhere(self.all_classes == label).item()
      self.indices[class_index, np.where(np.isnan(self.indices[class_index]))[0][0]] = i
      self.num_samples_per_class[class_index] += 1


  def __iter__(self):
    '''
    Yield a batch of sample indices per episode.

    Yield:
      Tensor: Batch of sample indices per episode.

    '''
    batch_size = self.num_classes * self.num_samples

    for _ in range(self.episodes):
      batch = torch.LongTensor(batch_size)
      class_indices = torch.randperm(self.total_classes)[:self.num_classes]

      for i, k in enumerate(self.all_classes[class_indices]):
        class_index    = torch.arange(self.total_classes).long()[self.all_classes == k].item()
        sample_indices = torch.randperm(self.num_samples_per_class[class_index])[:self.num_samples]
        batch[self.num_samples * i : self.num_samples * (i + 1)] = self.indices[class_index][sample_indices]

      yield batch[torch.randperm(len(batch))]


  def __len__(self) -> int:
    '''
    Return the number of episodes per epoch.

    Return:
      int: Number of episodes per epoch.

    '''
    return self.episodes

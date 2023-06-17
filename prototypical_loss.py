import torch
from torch.nn import functional as F
from torch.nn.modules import Module

# Define the prototypical loss function class
class PrototypicalLoss(Module):
  '''
  Loss class deriving from Module for the prototypical loss function defined below.

  '''
  def __init__(self):
    super(PrototypicalLoss, self).__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


  def forward(self, inputs: torch.Tensor, targets: torch.Tensor, num_support: int) -> tuple:
    '''
    Forward pass of the prototypical loss function.

    Args:
      inputs      (Tensor): Model outputs for a batch of samples.
      targets     (Tensor): Ground truth for the above batch of samples.
      num_support (int)   : Number of samples to keep in account when computing barycentres per class in the current classes.

    Returns:
      tuple[Tensor, Tensor]: Tuple containing the loss and the accuracy.

    '''
    return prototypical_loss(inputs, targets, num_support, self.device)


def prototypical_loss(inputs: torch.Tensor, targets: torch.Tensor, num_support: int, device: torch.device) -> tuple:
  '''
  Compute the barycentres by averaging the features of num_support samples per class in targets,
  compute then the distances from each sample's features to each one of the barycentres,
  compute the log probability for every num_query samples per class in the current classes appertaining to a class k,
  the negative log-likelihood loss and the accuracy are then computed and returned.

  Args:
    inputs      (Tensor): Model outputs for a batch of samples.
    targets     (Tensor): Ground truth for the above batch of samples.
    num_support (int)   : Number of samples to keep in account when computing barycentres per class in the current classes.
    device      (device): Device to calculate each loss and accuracy.

  Returns:
    tuple[Tensor, Tensor]: Tuple containing the loss and the accuracy.

  '''
  emb_samples, y = inputs.to(device), targets.to(device)

  classes   = torch.unique(y)
  num_query = y.eq(classes[0].item()).sum().item() - num_support
  y_query   = torch.arange(0, len(classes), 1 / num_query).long().to(device)

  # Make prototypes
  support_indices = torch.stack(list(map(lambda k: y.eq(k).nonzero()[:num_support].squeeze(dim = 1), classes)))
  prototypes      = torch.stack([emb_samples[indices_k].mean(dim = 0) for indices_k in support_indices])

  # Make embedded query samples
  query_indices     = torch.stack(list(map(lambda k: y.eq(k).nonzero()[num_support:], classes))).view(-1)
  emb_query_samples = emb_samples[query_indices]

  # Calculate Euclidean distances
  distances = euclidean_distances(emb_query_samples, prototypes)

  # Calculate negative log-likelihood loss and accuracy
  log_probs = F.log_softmax(-distances, dim = 1)
  y_preds   = log_probs.argmax(dim = 1)
  loss      = F.nll_loss(log_probs, y_query)
  accuracy  = y_preds.eq(y_query.squeeze()).float().mean()

  return loss, accuracy


def euclidean_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  '''
  Calculate the Euclidean distances between two tensors.

  Args:
    x (Tensor): Embedded query samples tensor. (N, D)
    y (Tensor): Prototypes tensor. (M, D)

  Return:
    Tensor: Euclidean distances tensor. (N, M)

  '''
  # x: (N, D)
  # y: (M, D)
  N = x.size(dim = 0)
  M = y.size(dim = 0)
  D = x.size(dim = 1)

  if D != y.size(dim = 1):
    raise Exception

  x = x.unsqueeze(dim = 1).expand(N, M, D)
  y = y.unsqueeze(dim = 0).expand(N, M, D)

  return torch.pow(x - y, 2).sum(dim = 2)

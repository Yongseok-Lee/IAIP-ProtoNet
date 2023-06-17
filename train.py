import argparse
import numpy as np
import torch
from torch import optim
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from root_dataset import RootDataset
from prototypical_batch_sampler import PrototypicalBatchSampler
from protonet import ProtoNet
from prototypical_loss import PrototypicalLoss

def main(args: argparse.Namespace) -> None:
  '''
  Main function for the training process.

  Args:
    args (Namespace): Namespace object of the command line arguments entered when running this program.

  '''
  ## 1) Data loading and preprocessing
  preprocess = T.Compose([
    T.ToTensor(),                           # Range of each pixel: [ 0, 1]
    T.Normalize(mean = (0.5), std = (0.5))  # Range of each pixel: [-1, 1]
  ])

  print('Loading data...', end = '')
  train_dataset = RootDataset(mode = 'train', transform = preprocess)
  test_dataset  = RootDataset(mode = 'test' , transform = preprocess)

  # Split the original train dataset into train dataset and validation dataset
  train_dataset, valid_dataset = random_split(train_dataset, [0.8, 0.2])

  # Get the labels of each dataset
  _, train_labels = zip(*train_dataset)
  _, valid_labels = zip(*valid_dataset)
  _, test_labels  = zip(*test_dataset)

  # Make the dataloaders of each dataset
  train_loader = DataLoader(train_dataset, pin_memory = True if torch.cuda.is_available() else False,
                            batch_sampler = PrototypicalBatchSampler(train_labels, args.episodes, num_classes = 5, num_samples = args.support + args.query_train))
  valid_loader = DataLoader(valid_dataset, pin_memory = True if torch.cuda.is_available() else False,
                            batch_sampler = PrototypicalBatchSampler(valid_labels, args.episodes, num_classes = 2, num_samples = args.support + args.query_valid))
  test_loader  = DataLoader(test_dataset , pin_memory = True if torch.cuda.is_available() else False,
                            batch_sampler = PrototypicalBatchSampler(test_labels , args.episodes, num_classes = 2, num_samples = args.support + args.query_valid))
  print(' Done!')

  ## 2) Initialize the Prototypical Networks model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model     = ProtoNet(in_channels = 1, num_filters = 64, out_channels = 64).to(device)
  criterion = PrototypicalLoss().to(device)

  optimizer    = optim.Adam(model.parameters(), args.learning_rate)
  lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)

  ## 3) Training loop
  writer = SummaryWriter()

  best_accuracy = 0

  for epoch in range(1, args.epochs + 1):
    print(f'Epoch {epoch}/{args.epochs}')

    train(model, train_loader, epoch, args.support, criterion, optimizer, writer, device)
    accuracy = validate(model, valid_loader, epoch, args.support, criterion, writer, device)

    if accuracy >= best_accuracy:
      best_accuracy = accuracy
      writer.add_scalar('Best Accuracy', best_accuracy, epoch)

    lr_scheduler.step()

  print(f'Best Accuracy: {100 * best_accuracy:.2f}%')

  test(model, test_loader, epoch, args.support, criterion, writer, device)

  writer.close()


def train(model: ProtoNet, train_loader: DataLoader, epoch: int, num_support: int, criterion: torch.nn.Module,
          optimizer: optim.Optimizer, writer: SummaryWriter, device: torch.device) -> None:
  '''
  Function for the training loop.

  Args:
    model        (ProtoNet)     : The Prototypical Networks model.
    train_loader (DataLoader)   : DataLoader for the training dataset.
		epoch        (int)          : Current count number of epoch.
    num_support  (int)          : Number of support samples per class per episode.
    criterion    (Module)       : Loss and accuracy criterion.
    optimizer    (Optimizer)    : Optimizer for model parameters.
		writer       (SummaryWriter): SummaryWriter for TensorBoard.
    device       (device)       : Device to run the training on.

  '''
  episodes    = len(train_loader)
  total_epoch = episodes * (epoch - 1)

  model.train()

  losses, accuracies = [], []

  for batch_index, (images, labels) in enumerate(train_loader):
    episode = batch_index + 1
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    loss, accuracy = criterion(outputs, labels, num_support)
    losses.append(loss.item())
    accuracies.append(accuracy.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('Loss/Train'    , loss.item()    , total_epoch + episode)
    writer.add_scalar('Accuracy/Train', accuracy.item(), total_epoch + episode)

    if episode % 10 == 0:
      print(f'episode {episode}\tLoss: {loss.item():.4f}\tAccuracy: {100 * accuracy.item():.2f}%')

  average_loss     = np.mean(losses)
  average_accuracy = np.mean(accuracies)

  writer.add_scalar('Average Loss/Train'    , average_loss    , total_epoch + episodes)
  writer.add_scalar('Average Accuracy/Train', average_accuracy, total_epoch + episodes)
  print(f'Average = Loss/Train: {average_loss:.4f}\tAccuracy/Train: {100 * average_accuracy:.2f}%')


def validate(model: ProtoNet, valid_loader: DataLoader, epoch: int, num_support: int, criterion: torch.nn.Module, writer: SummaryWriter, device: torch.device) -> None:
  '''
  Function for validating the trained model

  Args:
    model        (ProtoNet)     : The Prototypical Networks model.
    valid_loader (DataLoader)   : DataLoader for the validation dataset.
		epoch        (int)          : Current count number of epoch.
    num_support  (int)          : Number of support samples per class per episode.
    criterion    (Module)       : Loss and accuracy criterion.
		writer       (SummaryWriter): SummaryWriter for TensorBoard.
    device       (device)       : Device to run the evaluation on.

  '''
  episodes    = len(valid_loader)
  total_epoch = episodes * (epoch - 1)

  model.eval()

  losses, accuracies = [], []

  with torch.no_grad():
    for batch_index, (images, labels) in enumerate(valid_loader):
      episode = batch_index + 1
      images, labels = images.to(device), labels.to(device)

      outputs = model(images)
      loss, accuracy = criterion(outputs, labels, num_support)
      losses.append(loss.item())
      accuracies.append(accuracy.item())

      writer.add_scalar('Loss/Validation'    , loss.item()    , total_epoch + episode)
      writer.add_scalar('Accuracy/Validation', accuracy.item(), total_epoch + episode)

      if episode % 10 == 0:
        print(f'episode {episode}\tLoss: {loss.item():.4f}\tAccuracy: {100 * accuracy.item():.2f}%')

  average_loss     = np.mean(losses)
  average_accuracy = np.mean(accuracies)

  writer.add_scalar('Average Loss/Validation'    , average_loss    , total_epoch + episodes)
  writer.add_scalar('Average Accuracy/Validation', average_accuracy, total_epoch + episodes)
  print(f'Average = Loss/Validation: {average_loss:.4f}\tAccuracy/Validation: {100 * average_accuracy:.2f}%')

  return average_accuracy


def test(model: ProtoNet, test_loader: DataLoader, epoch: int, num_support: int, criterion: torch.nn.Module, writer: SummaryWriter, device: torch.device) -> None:
  '''
  Function for testing the model's accuracy.

  Args:
    model       (ProtoNet)     : The Prototypical Networks model.
    test_loader (DataLoader)   : DataLoader for the test dataset.
		epoch       (int)          : Current count number of epoch.
    num_support (int)          : Number of support samples per class per episode.
    criterion   (Module)       : Loss and accuracy criterion.
		writer      (SummaryWriter): SummaryWriter for TensorBoard.
    device      (device)       : Device to run the evaluation on.

  '''
  model.eval()

  accuracies = []

  with torch.no_grad():
    for epoch in range(10):
      for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, accuracy = criterion(outputs, labels, num_support)
        accuracies.append(accuracy.item())

  average_accuracy, margin_of_error = confidence_interval(accuracies, 1.96)
  total_epoch = len(test_loader) * epoch

  writer.add_scalar('Accuracy/Test', average_accuracy, total_epoch)
  plus_minus = u'\u00B1'
  print(f'Accuracy/Test: ({100 * average_accuracy:.2f} {plus_minus} {100 * margin_of_error:.2f})%')


def confidence_interval(values, critical_value):
  sample_mean    = np.mean(values)
  standard_error = np.std(values) / np.sqrt(len(values))

  return sample_mean, critical_value * standard_error


if __name__ == '__main__':
  # Command line argument parsing
  parser = argparse.ArgumentParser()
  parser.add_argument('-i' , '--episodes'     , type = int  , default =    50, help = 'Number of episodes (iterations) per epoch. (default: 50)')
  parser.add_argument('-s' , '--support'      , type = int  , default =     5, help = 'Number of support samples per class per episode. (default: 5)')
  parser.add_argument('-qt', '--query_train'  , type = int  , default =     5, help = 'Number of query samples for training per class per episode. (default: 5)')
  parser.add_argument('-qv', '--query_valid'  , type = int  , default =    15, help = 'Number of query samples for validation per class per episode. (default: 15)')
  parser.add_argument('-lr', '--learning_rate', type = float, default = 0.001, help = 'Learning rate for the optimizer. (default: 0.001)')
  parser.add_argument('-st', '--step_size'    , type = int  , default =    10, help = 'Step size of StepLR learning rate scheduler. (default: 10)')
  parser.add_argument('-g' , '--gamma'        , type = float, default =   0.5, help = 'Gamma value of StepLR learning rate scheduler. (default: 0.5)')
  parser.add_argument('-e' , '--epochs'       , type = int  , default =    50, help = 'Number of training epochs. (default: 50)')
  args = parser.parse_args()

  # Start the training process
  main(args)

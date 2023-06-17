import torch
from torch import nn

# Define the Prototypical Networks model class
class ProtoNet(nn.Module):
  def __init__(self, in_channels: int, num_filters: int, out_channels: int):
    '''
    Initialize the Prototypical Networks model for few-shot learning.

    Args:
      in_channels  (int): Number of channels in the input image.
      num_filters  (int): Number of filters in the convolution layer.
      out_channels (int): Number of output channels.

    '''
    super(ProtoNet, self).__init__()
    self.encoder = nn.Sequential(
      conv_block(in_channels, num_filters),
      conv_block(num_filters, num_filters),
      conv_block(num_filters, num_filters),
      conv_block(num_filters, out_channels)
    )


  def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
    '''
    Forward pass of the Prototypical Networks model.

    Args:
      input_tensor (Tensor): Input tensor of shape (batch_size, in_channels, height, width).

    Returns:
      Tensor: Output tensor of shape (batch_size, out_channels).

    '''
    output_tensor = self.encoder(input_tensor)

    return output_tensor.view(output_tensor.size(dim = 0), -1)


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
  '''
  Create a convolutional block consisting of a sequence of operations.

  Args:
    in_channels  (int): Number of channels in the input image.
    out_channels (int): Number of channels in the feature map.

  Returns:
    Sequential: Sequential container of the operations.

  '''
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
    nn.BatchNorm2d(num_features = out_channels),
    nn.ReLU(inplace = True),
    nn.MaxPool2d(kernel_size = 2, stride = 2)
  )

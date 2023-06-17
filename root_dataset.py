import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from torch.utils.data import Dataset

# Define the root dataset class
class RootDataset(Dataset):
  def __init__(self, mode: str, transform: callable = None):
    '''
    Initialize the RootDataset.

    Args:
      mode      (str)               : Mode to select the root directory containing the image files.
      transform (callable, optianal): Optional data transformation to be applied to the images.

    '''
    self.root_dir = mode
    self.transform = transform

    self.abnormal_individuals = ['root1_220930', 'root2_220914', 'root2_220919',
                                 'root2_221005', 'root3_220919', 'root3_221121']
    self.image_files = self._get_image_files()
    self.earliest_datetimes = {}


  def _get_image_files(self) -> list:
    '''
    Get a list of image file paths in the root directory.

    Returns:
      list[str]: List of image file paths.

    '''
    image_files = []

    for root, _, files in os.walk(self.root_dir):
      for filename in files:
        if filename.endswith('.jpg') and filename not in self.abnormal_individuals:
          image_files.append(os.path.join(root, filename))

    return image_files


  def __len__(self) -> int:
    '''
    Get the total number of images in the dataset.

    Returns:
      int: Number of images.

    '''
    return len(self.image_files)


  def __getitem__(self, index: int) -> tuple:
    '''
    Get the image and its corresponding label at the given index.

    Args:
      index (int): Index of the image.

    Returns:
      tuple[Tensor, int]: Tuple containing the image and its label.

    '''
    image_path = self.image_files[index]

    # Load and preprocess the image
    raw_image       = Image.open(image_path)
    cropped_image   = raw_image.crop((100, 0, 240, 174))
    bgr_image       = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
    gray_image      = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

    if self.transform:
      image = self.transform(binary_image)

    # Get label of the image
    label = self._get_label(image_path)

    return image, label


  def _get_label(self, image_path: str) -> int:
    '''
    Get the label for the given image path based on the timestamp.

    Args:
      image_path (str): Path of the image.

    Returns:
      int: Label assigned based on the time interval corresponding to the image.

    '''
    filename = os.path.basename(image_path)
    datetime = self._get_datetime(filename)

    individual_dir = os.path.dirname(image_path)
    earliest_datetime = self._get_earliest_datetime(individual_dir)

    hours = (datetime - earliest_datetime).total_seconds() / 3600

    if hours <= 12:
      return 0
    elif hours <= 24:
      return 1
    elif hours <= 36:
      return 2
    elif hours <= 48:
      return 3
    elif hours <= 60:
      return 4
    elif hours <= 72:
      return 5
    else:
      return 6


  def _get_earliest_datetime(self, individual_dir: str) -> datetime:
    '''
    Get the earliest datetime among all the images in the directory.

    Args:
      individual_dir (str): Path fo the individual's directory.

    Returns:
      datetime: The earliest datetime or None if no images are found.

    '''
    if individual_dir in self.earliest_datetimes:
      return self.earliest_datetimes[individual_dir]

    earliest_datetime = None

    for _, _, files in os.walk(individual_dir):
      for filename in files:
        if filename.endswith('.jpg'):
          datetime = self._get_datetime(filename)

          if earliest_datetime is None or datetime < earliest_datetime:
            earliest_datetime = datetime

    self.earliest_datetimes[individual_dir] = earliest_datetime

    return earliest_datetime


  def _get_datetime(self, filename: str) -> datetime:
    '''
    Convert the timestamp from the filename into a datetime object.

    Args:
      filename (str): Image filename.

    Returns:
      datetime: The datetime object corresponding to the timestamp in the filename.

    '''
    timestamp = filename.split('_')[-1].split('.')[0].split('(')[0]

    return datetime.strptime(timestamp, '%y%m%d%H%M%S')

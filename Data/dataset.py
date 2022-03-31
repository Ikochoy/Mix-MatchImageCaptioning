
from .imageset import ImageSet

class DataSet:
  def __init__(self, images):
    self.train = ImageSet()
    self.validation = ImageSet()
  
  def add_training_image(self, image):
    self.train.add_image(image)
  
  def add_validation_image(self, image):
    self.validation.add_image(self)
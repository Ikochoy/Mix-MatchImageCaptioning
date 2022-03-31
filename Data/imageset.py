class ImageSet:
  def __init__(self, images={}):
    # Dictionary of images
    self.images = images
  
  def add_image(self,image_id, image):
    self.images[image_id] = image
  
  def get_image_by_id(self, image_id):
    return self.images[image_id]
  
  def get_image_list(self):
      pass



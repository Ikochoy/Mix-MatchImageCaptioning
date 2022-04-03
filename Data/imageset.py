class ImageSet:
  def __init__(self, images={}):
    # Dictionary of images
    self.images = images
    self.annotations = {}
    self.files = {}
  
  def add_image(self,image_id, image):
    self.images[image_id] = image
    self.annotations[image_id] = image.annotations
    self.files[image_id] = image.file_name
  
  def get_image_by_id(self, image_id):
    return self.images[image_id]
  
  def get_annotation_list(self):
    return list(self.annotations.values())

  def get_image_ids(self):
    return list(self.images.keys())
  
  def get_images(self):
    return list(self.files.values())



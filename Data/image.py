class Image:
  def __init__(self, image_id, annotations=[], file_name=None):
    self.image_id = image_id
    self.annotations = annotations
    self.file_name = file_name
  
  def add_annotation(self, annotation):
    self.annotations.append(annotation)

  def get_annotations(self):
    return self.annotations

  def __str__(self):
    idx = f"{self.image_id}\n" 
    annotations = "\n".join(self.annotations)
    file_url = f"{self.file_name}"
    return idx + annotations + file_url


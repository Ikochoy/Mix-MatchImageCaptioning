import torch.nn as nn
import torchvision.models as models 
import torch
from torchvision import transforms


class ImageEncoder(nn.Module):
  def __init__(self, choice, dropout=False, fine_tune=False):
    super(ImageEncoder, self).__init__()
    self.input = CROPPED_WIDTH
    # Detect if we have a GPU available
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.trained = False
    if dropout:
        self.dropout = nn.Dropout(p=0.5)
    else:
        self.dropout = None
    self.transform = transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if choice == "GoogLeNet":
      # GoogLeNet
      self.model_name = "GoogLeNet_Inception_v3"
      model_ft = models.inception_v3(pretrained=True)
     
    elif choice == "AlexNet":
      self.model_name = "AlexNet"
      model_ft = models.alexnet(pretrained=True)

    elif choice == "VGG-19":
      self.model_name = "VGG-19"
      model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
      
    self.model = model_ft.to(self.device)  
    layers = list(self.model.to(self.device).children())[:-2]
    self.model = nn.Sequential(*layers)


    # Freeze parameters so we don't backprop through them
    for param in self.model.parameters():
        param.requires_grad = False
    # Fine tune the last layers
    if fine_tune:
      for layer in list(self.model.children())[-2:]:
        for param in layer.parameters():
          param.requires_grad = True
      
  
    
  
  def forward(self, X):
    # think about the ensemble and dropout a bit more
    features = self.model(X)
    if self.dropout:
        features = self.dropout(features)
    return features

  def encode_image(self, image_path, normalize=False):
    """
    have to make this so that this can also process a lot of images at the same time
    """
    img = Image.open(image_path)
    img_processed = self.transform(img).unsqueeze(0)
    if torch.cuda.is_available():
      img_processed = img_processed.to('cuda')
      self.model.to('cuda')

    if not self.trained:
      print("Model has not been trained")
      return -1
    with torch.no_grad():
      output = self.model(img_processed)[0]
    if normalize:
      # add normalization code here
      output = torch.nn.functional.softmax(output, dim=0)
    return output